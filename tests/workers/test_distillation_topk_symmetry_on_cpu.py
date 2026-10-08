# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Regression guard for verl#6293.

The use_remove_padding=False branch of
FSDPEngineWithLMHead.prepare_model_outputs previously lacked the
distillation_use_topk handling that the use_remove_padding=True branch had,
so distillation outputs were silently dropped from model_output and the
downstream loss raised KeyError. This test invokes prepare_model_outputs on
a stub engine for both branches with distillation_use_topk=True and asserts
the distillation keys produced by logits_processor_func are propagated into
model_output as nested tensors in both cases.

``logprobs_from_logits`` is patched out: in CI environments where flash-attn
is installed, it dispatches to a Triton CrossEntropyLoss kernel that cannot
operate on CPU tensors. The substitute returns a dummy ``log_probs`` tensor
of the right shape, which is sufficient for this test — the contract under
test is the propagation of distillation keys, not the numerical correctness
of log-prob computation.
"""

import asyncio
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk as compute_fsdp_forward_kl_topk
from verl.trainer.distillation.losses import compute_forward_kl_topk as collect_forward_kl_topk_metrics
from verl.trainer.distillation.losses import compute_topk_loss, distillation_loss
from verl.utils import tensordict_utils as tu
from verl.utils.config import _validate_vopd_config, omega_conf_to_dataclass
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.rollout.replica import TokenOutput
from verl.workers.utils.padding import no_padding_2_padding

_VOCAB_SIZE = 8
_DISTILLATION_KEYS = ("distillation_losses", "student_mass", "overlap_count", "overlap_token_advantage")


def _make_engine_stub():
    """Bypass FSDPEngineWithLMHead.__init__; set only attributes that
    prepare_model_outputs touches in this test path (no SP, no fused kernels,
    no entropy)."""
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.use_ulysses_sp = False

    class _EngineCfg:
        entropy_checkpointing = False

    eng.engine_config = _EngineCfg()
    return eng


def _make_logits_processor(keys):
    """Fake top-k distillation processor: returns one (1, total_nnz) tensor per key.

    The real processor (verl/trainer/distillation/losses.py) returns
    student_logits.shape[:2]; we mimic that contract.
    """

    def _proc(student_logits, data):
        n = student_logits.shape[1]
        return {k: torch.full((1, n), float(i + 1)) for i, k in enumerate(keys)}

    return _proc


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("distillation_only", [False, True])
def test_distillation_outputs_emitted_in_both_padding_modes(use_remove_padding, distillation_only):
    """distillation_use_topk=True must populate distillation outputs into
    model_output regardless of use_remove_padding. See verl#6293.

    When distillation_only=True, log_probs must be omitted (supervised top-k path)."""
    bsz = 2
    seq_lengths_list = [3, 2]
    seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int64)
    total_nnz = int(seq_lengths.sum())

    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)]).to(torch.int64)

    flat_input_ids = torch.randint(0, _VOCAB_SIZE, (total_nnz,))
    input_ids_nested = torch.nested.nested_tensor_from_jagged(flat_input_ids, offsets=cu_seqlens)

    input_ids_rmpad_rolled = torch.randint(0, _VOCAB_SIZE, (total_nnz,))

    class _Output:
        pass

    output = _Output()

    if use_remove_padding:
        # True branch: output.logits shape (1, total_nnz, V), squeeze(0) -> (total_nnz, V).
        output.logits = torch.randn(1, total_nnz, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature_rmpad": torch.ones(total_nnz),
            # No SP and no static pad_to_length here, so nothing to trim off the packed tail.
            "pad_size": 0,
        }
    else:
        # False branch: output.logits shape (bsz, max_seqlen, V).
        max_seqlen = max(seq_lengths_list)
        output.logits = torch.randn(bsz, max_seqlen, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature": torch.ones(bsz),
        }

    micro_batch = TensorDict({"input_ids": input_ids_nested}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        distillation_only=distillation_only,
        max_response_length=max(seq_lengths_list),
    )

    eng = _make_engine_stub()

    # Patch logprobs_from_logits because flash-attn's Triton CrossEntropyLoss
    # cannot operate on CPU tensors. The shape is what downstream code asserts
    # against (v.shape == log_probs.shape), and prepare_model_outputs reduces
    # both branches to a (total_nnz,) log_probs over the rmpad'ed logits.
    with patch(
        "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits",
        return_value=torch.zeros(total_nnz),
    ):
        model_output = FSDPEngineWithLMHead.prepare_model_outputs(
            eng,
            output=output,
            output_args=output_args,
            micro_batch=micro_batch,
            logits_processor_func=_make_logits_processor(_DISTILLATION_KEYS),
        )

    if distillation_only:
        assert "log_probs" not in model_output, (
            f"log_probs should be omitted when distillation_only=True "
            f"(use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
        )
    else:
        assert "log_probs" in model_output, (
            f"log_probs missing (use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
        )

    for k in _DISTILLATION_KEYS:
        assert k in model_output, (
            f"Distillation key '{k}' missing from model_output "
            f"(use_remove_padding={use_remove_padding}); "
            f"keys: {list(model_output.keys())}"
        )
        assert model_output[k].is_nested, (
            f"Expected '{k}' to be a nested tensor (use_remove_padding={use_remove_padding}); "
            f"got {type(model_output[k])}"
        )


def _nested_from_rows(rows):
    values = torch.tensor(rows)
    offsets = torch.tensor([0, len(rows)], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def test_forward_kl_topk_emits_overlap_metrics():
    logits = torch.tensor(
        [
            [0.0, 9.0, 8.0, 1.0, 0.0, 0.0],
            [8.0, 7.0, 0.0, 0.0, 9.0, 0.0],
            [9.0, 8.0, 7.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    teacher_ids = _nested_from_rows([[1, 2], [4, 5], [3, 4]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows(
        [
            [torch.log(torch.tensor(0.7)), torch.log(torch.tensor(0.2))],
            [torch.log(torch.tensor(0.6)), torch.log(torch.tensor(0.3))],
            [torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.4))],
        ]
    ).to(torch.float32)
    config = SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=None))

    output = compute_fsdp_forward_kl_topk(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=config,
        data_format="thd",
    )

    torch.testing.assert_close(output["overlap_count"], torch.tensor([[2, 1, 0]]))

    student_log_probs = torch.log_softmax(logits, dim=-1)
    gathered_student = torch.gather(student_log_probs, dim=-1, index=teacher_ids.values().unsqueeze(0))
    teacher_log_probs = teacher_logprobs.values().unsqueeze(0)
    token_adv = -(teacher_log_probs.exp() * (teacher_log_probs - gathered_student))
    expected_ota = torch.tensor(
        [[token_adv[0, 0].mean(), token_adv[0, 1, 0], 0.0]],
        dtype=output["overlap_token_advantage"].dtype,
    )
    torch.testing.assert_close(output["overlap_token_advantage"], expected_ota)


def test_forward_kl_topk_metric_aggregation_for_overlap_outputs():
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "distillation_losses": torch.tensor([0.1, 0.2, 0.3]),
        "student_mass": torch.tensor([0.9, 0.8, 0.7]),
        "teacher_mass": torch.tensor([0.95, 0.85, 0.75]),
        "overlap_count": torch.tensor([2, 1, 0]),
        "overlap_token_advantage": torch.tensor([-0.2, -0.4, 0.0]),
    }
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2))

    _, metrics = collect_forward_kl_topk_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )

    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.75)
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(-0.3)


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_vopd_baseline_through_fsdp_output_processor(use_remove_padding, monkeypatch):
    from transformers.modeling_outputs import CausalLMOutput

    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "1")
    offsets = torch.tensor([0, 3, 5])
    tokens = torch.tensor([1, 2, 3, 1, 4])
    logits = torch.randn(1, 5, _VOCAB_SIZE, requires_grad=True)
    teacher_logps = torch.randn_like(logits).log_softmax(-1)
    labels = tokens.roll(-1)
    support = logits.detach().topk(2, -1).indices
    teacher_ids = torch.cat([support, labels.view(1, -1, 1)], -1)
    teacher_logps = teacher_logps.gather(-1, teacher_ids)
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor_from_jagged(tokens, offsets),
            "teacher_ids": torch.nested.nested_tensor_from_jagged(teacher_ids.squeeze(0), offsets),
            "teacher_logprobs": torch.nested.nested_tensor_from_jagged(teacher_logps.squeeze(0), offsets),
        },
        batch_size=[],
    )
    tu.assign_non_tensor(
        data,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        distillation_only=False,
    )
    actor = ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True)
    config = DistillationConfig(distillation_loss=vopd_loss_config(2))

    def processor(student_logits, data):
        return compute_topk_loss(actor, config, data, student_logits, "thd")

    expected = processor(logits, data)
    if use_remove_padding:
        output = CausalLMOutput(logits=logits)
        args = {"input_ids_rmpad_rolled": labels, "temperature_rmpad": torch.ones(5), "pad_size": 0}
    else:
        padded = torch.stack([logits[0, :3], torch.cat([logits[0, 3:], torch.zeros(1, _VOCAB_SIZE)])])
        output = CausalLMOutput(logits=padded)
        args = {"input_ids_rmpad_rolled": labels, "temperature": torch.ones(2)}
    model_output = _make_engine_stub().prepare_model_outputs(output, args, data, processor)
    for key, value in expected.items():
        torch.testing.assert_close(model_output[key].values(), value.squeeze(0))
        assert not model_output[key].requires_grad
    reference_logps = logits.log_softmax(-1).gather(-1, labels.view(1, -1, 1)).reshape(-1)
    torch.testing.assert_close(model_output["log_probs"].values(), reference_logps)
    model_output["log_probs"].values().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def vopd_loss_config(topk=3):
    return DistillationLossConfig(
        loss_mode="vopd_topk",
        topk=topk,
        use_policy_gradient=True,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
        use_task_rewards=False,
    )


def nested(values, offsets):
    return torch.nested.nested_tensor_from_jagged(values, offsets)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_vopd_pipeline_gradient(k):
    torch.manual_seed(8)
    offsets = torch.tensor([0, 5, 9])
    tokens = torch.tensor([1, 3, 2, 5, 6, 1, 4, 2, 3])
    labels = tokens.roll(-1)
    student = torch.randn(1, 9, 7, requires_grad=True)
    teacher = torch.randn(1, 9, 7, requires_grad=True)
    log_p, log_q = student.log_softmax(-1), teacher.log_softmax(-1)
    support = student.detach().topk(k, -1).indices
    ids = torch.cat([support, labels.view(1, -1, 1)], -1)
    scores = log_q.gather(-1, ids)
    bp, bq = log_p.gather(-1, support).log_softmax(-1), log_q.gather(-1, support).log_softmax(-1)
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
    config = DistillationConfig(distillation_loss=vopd_loss_config(k))
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
    loss, _ = distillation_loss(actor, config, model_output, data)
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


@pytest.fixture
def teacher_manager(vopd_config):
    client = AsyncMock()
    return AsyncTeacherLLMServerManager(vopd_config, {"default": client}), client


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


@pytest.fixture
def vopd_config():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "config"
    overrides = [
        "distillation.enabled=True",
        "distillation.n_gpus_per_node=1",
        "distillation.nnodes=1",
        "distillation.teacher_models.teacher_model.model_path=test-model",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
        "distillation.teacher_models.teacher_model.inference.name=vllm",
        "+distillation.teacher_models.teacher_model.inference.logprobs_mode=raw_logprobs",
        "distillation.distillation_loss.loss_mode=vopd_topk",
        "distillation.distillation_loss.topk=2",
        "distillation.distillation_loss.use_policy_gradient=True",
        "distillation.distillation_loss.use_task_rewards=False",
        "distillation.distillation_loss.loss_max_clamp=null",
        "distillation.distillation_loss.log_prob_min_clamp=null",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.topk_log_probs=2",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    return config


def test_composed_vopd_hydra_recipe(vopd_config):
    _validate_vopd_config(vopd_config)
    resolved = omega_conf_to_dataclass(vopd_config.distillation)
    assert resolved.distillation_loss.loss_mode == "vopd_topk"
    assert resolved.teacher_models["default"].inference.logprobs_mode == "raw_logprobs"
    for key, value in [("do_sample", False), ("temperature", 0.7), ("top_p", 0.9), ("top_k", 5)]:
        invalid = OmegaConf.create(OmegaConf.to_container(vopd_config, resolve=False))
        invalid.actor_rollout_ref.rollout[key] = value
        with pytest.raises(ValueError, match="do_sample=True"):
            _validate_vopd_config(invalid)


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
