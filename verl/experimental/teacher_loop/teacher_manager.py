# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    # Temperature has no effect on prompt_logprobs: the teacher performs a forward pass over
    # existing tokens (no sampling). Always use temperature=1.0 regardless of the config value.
    # The default distillation.yaml copies the student rollout temperature via Hydra interpolation
    # (temperature: ${oc.select:actor_rollout_ref.rollout.temperature}), which causes a spurious
    # crash when rollout.temperature != 1.0.
    if teacher_model_config.inference.temperature != 1.0:
        logger.warning(
            "Teacher inference temperature is set to %.1f, but temperature has no effect "
            "on prompt_logprobs (forward pass only). Using temperature=1.0.",
            teacher_model_config.inference.temperature,
        )
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    if distillation_loss_config.loss_mode == "vopd_full":
        num_logprobs = -1
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
        "detokenize": False,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client
        # Bound selective scoring across all trajectories handled by this manager.
        self._vopd_semaphore = asyncio.Semaphore(8)

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        mm_processor_output: Optional[list[dict[str, Any]]] = None,
        routing_key: Optional[str] = None,
        student_topk_ids: Optional[list[list[int]]] = None,
        prompt_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        if self.distillation_loss_config.loss_mode == "vopd_topk":
            if multi_modal_data:
                raise NotImplementedError("vOPD selective prefix scoring currently supports text-only sequences.")
            return await self._score_student_support(client, sequence_ids, student_topk_ids, prompt_length)
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_output=mm_processor_output,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs

    async def _score_student_support(
        self,
        client: LLMServerClient,
        sequence_ids: list[int],
        student_topk_ids: Optional[list[list[int]]],
        prompt_length: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score each student support with a one-token vLLM request on its prefix.

        vLLM 0.29 supports selected output logprobs, but not per-position selected
        prompt logprobs. Prefix requests bound score transfer to O(T*k), at the
        cost of T prefills. Prefix caching on the teacher can amortize those prefills.
        """
        k = self.distillation_loss_config.topk
        if prompt_length is None or not 1 <= prompt_length <= len(sequence_ids) or student_topk_ids is None:
            raise ValueError("vOPD top-k requires the student rollout's response_topk_ids and prompt length.")
        response_length = len(sequence_ids) - prompt_length
        if response_length == 0:
            ids = torch.cat([torch.arange(k), torch.zeros(1, dtype=torch.long)]).repeat(len(sequence_ids), 1)
            return ids.to(torch.int32), torch.zeros(len(sequence_ids), k + 1)
        support = torch.as_tensor(student_topk_ids, dtype=torch.long)
        if support.shape != (response_length, k):
            raise ValueError(f"Expected student top-k shape {(response_length, k)}, got {tuple(support.shape)}.")
        if (support.sort(-1).values.diff(dim=-1) == 0).any():
            raise ValueError("Student top-k support must contain k distinct tokens at each position.")
        # Dummy prompt/last rows are masked out. Give them a valid distinct support.
        ids = torch.cat([torch.arange(k), torch.zeros(1, dtype=torch.long)]).repeat(len(sequence_ids), 1)
        scores = torch.zeros(len(sequence_ids), k + 1)

        async def score_position(index):
            position = prompt_length + index
            requested = support[index].tolist() + [sequence_ids[position]]
            # The sampled token may already be in the baseline support. Query it once,
            # then restore a separate final column so it never changes that support.
            unique = list(dict.fromkeys(requested))
            async with self._vopd_semaphore:
                output = await client.generate(
                    request_id=uuid4().hex,
                    prompt_ids=sequence_ids[:position],
                    sampling_params={
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "detokenize": False,
                        "logprob_token_ids": unique,
                    },
                )
            returned = output.extra_fields["requested_token_logprobs"]
            if len(returned) != len(unique):
                raise ValueError("Teacher did not return all requested token scores.")
            by_id = dict(zip(unique, returned, strict=True))
            ids[position - 1] = torch.tensor(requested)
            scores[position - 1] = torch.tensor([by_id[token] for token in requested])

        # A fixed worker count also avoids creating one task per response token.
        positions = iter(range(response_length))

        async def score_positions():
            for index in positions:
                await score_position(index)

        tasks = [asyncio.create_task(score_positions()) for _ in range(min(8, response_length))]
        try:
            await asyncio.gather(*tasks)
        finally:
            # A failed/cancelled trajectory must not leave teacher requests running.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return ids.to(torch.int32), scores
