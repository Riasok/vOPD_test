# Copyright 2026 Individual Contributor: Riasok
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
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager, _get_teacher_sampling_params
from verl.trainer.distillation.losses import compute_topk_loss, distillation_loss
from verl.utils.config import _validate_vopd_config
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig, DistillationTeacherModelConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.utils.padding import no_padding_2_padding


def loss_config(mode, topk=3):
    return DistillationLossConfig(
        loss_mode=mode,
        topk=topk,
        use_policy_gradient=True,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
        use_task_rewards=False,
    )


def nested(values, offsets):
    return torch.nested.nested_tensor_from_jagged(values, offsets)


@pytest.mark.parametrize("mode", ["vopd_topk", "vopd_full"])
@pytest.mark.parametrize("k", [1, 3, 7])
def test_vopd_pipeline_gradient(mode, k):
    torch.manual_seed(8)
    offsets = torch.tensor([0, 5, 9])
    tokens = torch.tensor([1, 3, 2, 5, 6, 1, 4, 2, 3])
    labels = tokens.roll(-1)
    student = torch.randn(1, 9, 7, requires_grad=True)
    teacher = torch.randn(1, 9, 7, requires_grad=True)
    log_p, log_q = student.log_softmax(-1), teacher.log_softmax(-1)
    support = student.detach().topk(k, -1).indices
    if mode == "vopd_topk":
        ids = torch.cat([support, labels.view(1, -1, 1)], -1)
        scores = log_q.gather(-1, ids)
        bp, bq = log_p.gather(-1, support).log_softmax(-1), log_q.gather(-1, support).log_softmax(-1)
    else:
        ids = torch.arange(7).expand(1, 9, 7)
        scores = log_q
        bp, bq = log_p, log_q
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    data = TensorDict(
        {
            "input_ids": nested(tokens, offsets),
            "prompts": nested(torch.tensor([1, 3, 1, 4]), torch.tensor([0, 2, 4])),
            "responses": nested(torch.tensor([2, 5, 6, 2, 3]), torch.tensor([0, 3, 5])),
            "teacher_ids": nested(ids.squeeze(0), offsets),
            "teacher_logprobs": nested(scores.squeeze(0), offsets),
            "response_mask": mask,
            "dp_size": 1,
            "batch_num_tokens": mask.sum(),
            "global_batch_size": 2,
        },
        batch_size=[],
    )
    config = DistillationConfig(distillation_loss=loss_config(mode, k))
    actor = ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True, loss_agg_mode="token-mean")
    result = compute_topk_loss(actor, config, data, student, "thd")
    expected_baseline = (bp.exp() * (bp - bq)).sum(-1)
    torch.testing.assert_close(result["vopd_baseline"], expected_baseline)
    assert all(not value.requires_grad for value in result.values())
    sample_logps = log_p.gather(-1, labels.view(1, -1, 1)).squeeze(0).squeeze(-1)
    model_output = {key: nested(value.squeeze(0), offsets) for key, value in result.items()}
    model_output["log_probs"] = nested(sample_logps, offsets)
    action_p = no_padding_2_padding(model_output["log_probs"], data)
    data["old_log_probs"] = action_p.detach()
    loss, metrics = distillation_loss(actor, config, model_output, data)
    # PPO's ratio surrogate has the same gradient as -A*log(p) at current == old.
    action_q = no_padding_2_padding(nested(log_q.gather(-1, labels.view(1, -1, 1)).view(-1), offsets), data)
    base = no_padding_2_padding(nested(expected_baseline.view(-1), offsets), data)
    reference = ((action_p - action_q - base).detach() * action_p * mask).sum() / mask.sum()
    grad = torch.autograd.grad(loss, student, retain_graph=True)[0]
    torch.testing.assert_close(
        grad, torch.autograd.grad(reference, student, retain_graph=True)[0], atol=1e-6, rtol=1e-5
    )
    loss.backward()
    assert teacher.grad is None
    assert "distillation/vopd_baseline" in metrics


@pytest.mark.parametrize(
    "bad",
    [
        {"use_policy_gradient": False},
        {"loss_max_clamp": 10.0},
        {"log_prob_min_clamp": -10.0},
        {"topk": 0},
        {"topk": 128},
    ],
)
def test_rejects_invalid_loss_settings(bad):
    kwargs = dict(loss_mode="vopd_topk", topk=3, use_policy_gradient=True, loss_max_clamp=None, log_prob_min_clamp=None)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        DistillationLossConfig(**kwargs)


def test_full_vocabulary_teacher_sampling():
    teacher = DistillationTeacherModelConfig(inference=RolloutConfig(name="vllm", temperature=1))
    assert _get_teacher_sampling_params(teacher, loss_config("vopd_full"))["prompt_logprobs"] == -1


@pytest.fixture
def teacher_manager():
    config = OmegaConf.create(
        {
            "distillation": {
                "_target_": "verl.workers.config.DistillationConfig",
                "enabled": True,
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "teacher_models": {
                    "teacher_model": {
                        "_target_": "verl.workers.config.DistillationTeacherModelConfig",
                        "model_path": "test-model",
                        "inference": {
                            "_target_": "verl.workers.config.RolloutConfig",
                            "name": "vllm",
                            "tensor_model_parallel_size": 1,
                            "logprobs_mode": "raw_logprobs",
                        },
                    }
                },
                "distillation_loss": {
                    "_target_": "verl.workers.config.DistillationLossConfig",
                    "loss_mode": "vopd_topk",
                    "topk": 2,
                    "use_policy_gradient": True,
                    "loss_max_clamp": None,
                    "log_prob_min_clamp": None,
                },
            }
        }
    )
    client = AsyncMock()
    return AsyncTeacherLLMServerManager(config, {"default": client}), client


@pytest.mark.asyncio
async def test_teacher_scores_student_support_and_separate_sample_column(teacher_manager):
    # Only the remote client is mocked; routing and configuration are real.
    manager, client = teacher_manager

    async def generate(**kwargs):
        requested = kwargs["sampling_params"]["logprob_token_ids"]
        return TokenOutput(token_ids=[0], extra_fields={"requested_token_logprobs": [-float(i + 1) for i in requested]})

    client.generate.side_effect = generate
    ids, scores = await manager.compute_teacher_logprobs_single(
        sequence_ids=[1, 2, 3, 6], prompt_length=2, student_topk_ids=[[3, 4], [2, 4]]
    )
    assert ids.tolist()[1:3] == [[3, 4, 3], [2, 4, 6]]
    assert scores.tolist()[1:3] == [[-4, -5, -4], [-3, -5, -7]]
    requests = client.generate.call_args_list
    assert requests[0].kwargs["prompt_ids"] == [1, 2]
    assert requests[1].kwargs["prompt_ids"] == [1, 2, 3]
    assert requests[0].kwargs["sampling_params"]["logprob_token_ids"] == [3, 4]
    assert requests[1].kwargs["sampling_params"]["logprob_token_ids"] == [2, 4, 6]
    # Prefix and last-token rows are dummies; only correctly shifted response rows are scored.
    assert torch.count_nonzero(scores[[0, 3]]) == 0
    with pytest.raises(ValueError, match="distinct"):
        await manager.compute_teacher_logprobs_single(
            sequence_ids=[1, 2, 3, 6], prompt_length=2, student_topk_ids=[[3, 3], [2, 4]]
        )


def test_rejects_unsupported_actor_backend():
    config = OmegaConf.create(
        {
            "distillation": {"enabled": True, "distillation_loss": {"loss_mode": "vopd_topk"}},
            "actor_rollout_ref": {"actor": {"strategy": "megatron"}, "rollout": {}},
        }
    )
    with pytest.raises(ValueError, match="FSDP"):
        _validate_vopd_config(config)


@dataclass
class Logprob:
    # Wire-level vLLM result fixture; importing vLLM is unnecessary for parsing.
    logprob: float
    rank: int | None = None


@dataclass
class PromptOutput:
    prompt_logprobs: list


def test_full_vocabulary_wire_parsing_and_alignment():
    from verl.workers.rollout.utils import extract_prompt_logprobs

    output = PromptOutput(
        [None, {2: Logprob(-3), 0: Logprob(-1), 1: Logprob(-2)}, {1: Logprob(-4), 2: Logprob(-5), 0: Logprob(-6)}]
    )
    result = {}
    extract_prompt_logprobs(output, -1, result)
    assert result["prompt_ids"] == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    assert result["prompt_logprobs"] == [[-1, -2, -3], [-6, -4, -5], [0, 0, 0]]
    with pytest.raises(ValueError, match="contiguous"):
        extract_prompt_logprobs(PromptOutput([None, {0: Logprob(-1), 2: Logprob(-3)}]), -1, {})


def test_existing_topk_and_sampled_wire_parsing():
    from verl.workers.rollout.utils import extract_prompt_logprobs, extract_requested_token_logprobs

    output = PromptOutput([None, {4: Logprob(-5, 5), 0: Logprob(-1, 1), 2: Logprob(-2, 2)}])
    result = {}
    extract_prompt_logprobs(output, 2, result)
    assert result == {"prompt_ids": [[0, 2], [0, 0]], "prompt_logprobs": [[-1, -2], [0, 0]]}
    extract_prompt_logprobs(output, 0, result)
    assert result == {"prompt_ids": [[4], [0]], "prompt_logprobs": [[-5], [0]]}
    assert extract_requested_token_logprobs(output.prompt_logprobs[1], [2, 4]) == [-2, -5]


@pytest.mark.parametrize("mode", ["vopd_topk", "vopd_full"])
def test_composed_hydra_recipe(mode):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from verl.utils.config import omega_conf_to_dataclass

    config_dir = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"
    overrides = [
        "distillation.enabled=True",
        "distillation.n_gpus_per_node=1",
        "distillation.nnodes=1",
        "distillation.teacher_models.teacher_model.model_path=test-model",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
        "distillation.teacher_models.teacher_model.inference.name=vllm",
        "+distillation.teacher_models.teacher_model.inference.logprobs_mode=raw_logprobs",
        f"distillation.distillation_loss.loss_mode={mode}",
        "distillation.distillation_loss.topk=20",
        "distillation.distillation_loss.use_policy_gradient=True",
        "distillation.distillation_loss.use_task_rewards=False",
        "distillation.distillation_loss.loss_max_clamp=null",
        "distillation.distillation_loss.log_prob_min_clamp=null",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        f"actor_rollout_ref.rollout.topk_log_probs={20 if mode == 'vopd_topk' else 0}",
        "actor_rollout_ref.model.use_fused_kernels=False",
        "actor_rollout_ref.actor.use_fused_kernels=False",
        "actor_rollout_ref.actor.pad_to_length=False",
        "actor_rollout_ref.actor.ppo_epochs=1",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    _validate_vopd_config(config)
    for key, value in [("do_sample", False), ("temperature", 0.7), ("top_p", 0.9), ("top_k", 5)]:
        invalid = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        invalid.actor_rollout_ref.rollout[key] = value
        with pytest.raises(ValueError, match="do_sample=True"):
            _validate_vopd_config(invalid)
    resolved = omega_conf_to_dataclass(config.distillation)
    assert resolved.distillation_loss.loss_mode == mode
    teacher = resolved.teacher_models["default"].inference
    assert teacher.logprobs_mode == "raw_logprobs"
    assert teacher.engine_kwargs["vllm"]["max_logprobs"] == (-1 if mode == "vopd_full" else 20)


@pytest.mark.asyncio
async def test_teacher_concurrency_is_bounded_across_trajectories(teacher_manager):
    manager, client = teacher_manager
    active = peak = 0

    async def generate(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.001)
            requested = kwargs["sampling_params"]["logprob_token_ids"]
            return TokenOutput(token_ids=[0], extra_fields={"requested_token_logprobs": [-1.0] * len(requested)})
        finally:
            active -= 1

    client.generate.side_effect = generate
    await asyncio.gather(
        *(
            manager.compute_teacher_logprobs_single([1] * 21, prompt_length=1, student_topk_ids=[[1, 2]] * 20)
            for _ in range(3)
        )
    )
    assert client.generate.await_count == 60
    assert peak == 8
    assert active == 0


@pytest.mark.asyncio
async def test_failed_teacher_request_cancels_remaining_work(teacher_manager):
    manager, client = teacher_manager
    active = 0
    started = asyncio.Event()

    async def generate(**kwargs):
        nonlocal active
        active += 1
        if active == 8:
            started.set()
        try:
            await started.wait()
            if len(kwargs["prompt_ids"]) == 1:
                raise RuntimeError("remote teacher failed")
            await asyncio.Event().wait()
        finally:
            active -= 1

    client.generate.side_effect = generate
    with pytest.raises(RuntimeError, match="remote teacher failed"):
        await asyncio.wait_for(
            manager.compute_teacher_logprobs_single([1] * 21, prompt_length=1, student_topk_ids=[[1, 2]] * 20),
            timeout=5,
        )
    assert active == 0


@pytest.mark.parametrize("chunk_size", [1, 17, 256, 300])
def test_full_vocabulary_baseline_chunk_boundary(chunk_size):
    from verl.trainer.distillation.fsdp.losses import compute_vopd_baseline

    torch.manual_seed(13)
    student = torch.randn(1, 270, 7, requires_grad=True)
    teacher = torch.randn_like(student).log_softmax(-1)
    offsets = torch.tensor([0, 270])
    labels = torch.arange(270) % 7
    data = TensorDict(
        {
            "input_ids": nested(labels, offsets),
            "teacher_logprobs": nested(teacher.squeeze(0), offsets),
        },
        batch_size=[],
    )
    result = compute_vopd_baseline(
        student, data, DistillationConfig(distillation_loss=loss_config("vopd_full")), chunk_size
    )
    log_probs = student.log_softmax(-1)
    torch.testing.assert_close(result["vopd_baseline"], (log_probs.exp() * (log_probs - teacher)).sum(-1))
    assert not result["vopd_baseline"].requires_grad
