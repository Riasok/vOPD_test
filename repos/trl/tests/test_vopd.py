# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from trl import DistillationConfig, DistillationTrainer
from trl.trainer.distillation_trainer import _chunked_divergence_loss


@pytest.mark.parametrize("top_k", [0, 1, 3, 7, 100])
@pytest.mark.parametrize("chunk_size", [1, 4, 20])
def test_vopd_gradient_matches_exact_action_expectation(top_k, chunk_size):
    # Enumerate every action at one fixed context. Its probability-weighted vOPD
    # gradient must equal the exact reverse-KL gradient, for ANY detached baseline.
    torch.manual_seed(4)
    vocab = 7
    student = torch.randn(1, 1, vocab, requires_grad=True)
    teacher = torch.randn(1, 1, vocab, requires_grad=True)
    weights = torch.eye(vocab, requires_grad=True)
    teacher_weights = torch.eye(vocab, requires_grad=True)
    probabilities = student.detach().softmax(-1).reshape(-1)
    expected_gradient = torch.zeros_like(student)
    for action in range(vocab):
        loss, _, _ = _chunked_divergence_loss(
            student,
            teacher,
            weights,
            teacher_weights,
            torch.ones(1, 1),
            1.0,
            chunk_size,
            completion_ids=torch.tensor([[action]]),
            loss_type="vopd",
            vopd_top_k=top_k,
        )
        grad = torch.autograd.grad(loss, student, retain_graph=True)[0]
        expected_gradient += probabilities[action] * grad
    p = student.log_softmax(-1)
    q = teacher.detach().log_softmax(-1)
    exact = (p.exp() * (p - q)).sum()
    torch.testing.assert_close(expected_gradient, torch.autograd.grad(exact, student)[0], atol=2e-6, rtol=2e-5)
    loss.backward()
    assert teacher.grad is None and teacher_weights.grad is None


@pytest.mark.parametrize("top_k", [0, 1, 3, 7, 100])
def test_vopd_chunk_mask_and_reference(top_k):
    torch.manual_seed(10)
    student = torch.randn(2, 4, 7, requires_grad=True)
    teacher = torch.randn(2, 4, 7, requires_grad=True)
    ids = torch.randint(7, (2, 4))
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]])
    loss, _, n_valid = _chunked_divergence_loss(
        student,
        teacher,
        torch.eye(7),
        torch.eye(7),
        mask,
        1.0,
        3,
        num_items_in_batch=10,
        temperature=0.7,
        completion_ids=ids,
        loss_type="vopd",
        vopd_top_k=top_k,
    )
    p = (student / 0.7).log_softmax(-1)
    q = (teacher / 0.7).log_softmax(-1)
    p_sample = p.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    q_sample = q.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    if top_k:
        support = student.topk(min(top_k, 7), -1).indices
        p_base = p.gather(-1, support).log_softmax(-1)
        q_base = q.gather(-1, support).log_softmax(-1)
    else:
        p_base, q_base = p, q
    kl = (p_base.exp() * (p_base - q_base)).sum(-1)
    expected = ((p_sample - q_sample - kl).detach() * p_sample * mask).sum() / 10
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.autograd.grad(expected, student)[0])
    assert n_valid == mask.sum()


@pytest.mark.parametrize("top_k", [0, 3])
def test_vopd_all_masked(top_k):
    student = torch.randn(1, 3, 7, requires_grad=True)
    loss, _, _ = _chunked_divergence_loss(
        student,
        torch.randn_like(student),
        torch.eye(7),
        torch.eye(7),
        torch.zeros(1, 3),
        1.0,
        2,
        completion_ids=torch.zeros(1, 3, dtype=torch.long),
        loss_type="vopd",
        vopd_top_k=top_k,
    )
    loss.backward()
    assert loss == 0
    assert torch.count_nonzero(student.grad) == 0


@pytest.mark.parametrize("top_k", [0, 3])
def test_vopd_real_trainer_step(tmp_path, top_k):
    torch.manual_seed(5)
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "a": 3, "b": 4, "c": 5, "d": 6}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    model_config = GPT2Config(
        vocab_size=7, n_embd=16, n_layer=1, n_head=2, n_positions=32, bos_token_id=1, eos_token_id=1, pad_token_id=0
    )
    student = GPT2LMHeadModel(model_config)
    teacher = GPT2LMHeadModel(model_config)
    teacher_before = {k: v.detach().clone() for k, v in teacher.named_parameters()}
    student_before = student.transformer.h[0].attn.c_attn.weight.detach().clone()
    args = DistillationConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        report_to="none",
        loss_type="vopd",
        vopd_top_k=top_k,
        max_steps=1,
        per_device_train_batch_size=2,
        max_completion_length=3,
        save_strategy="no",
    )
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_dict({"prompt": ["a b", "b c"]}),
    )
    # Exercise the causal completion alignment and tool mask against a full model forward.
    inputs = {
        "prompt_ids": torch.tensor([[3, 4], [4, 5]]),
        "prompt_mask": torch.ones(2, 2, dtype=torch.long),
        "completion_ids": torch.tensor([[5, 6, 1], [3, 1, 0]]),
        "completion_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "tool_mask": torch.tensor([[1, 0, 1], [1, 1, 0]]),
    }
    student.eval()
    loss = trainer.compute_loss(student, inputs)
    ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], -1)
    attention = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], -1)
    p = student(ids, attention_mask=attention).logits[:, 1:-1].float().log_softmax(-1)
    with torch.no_grad():
        q = teacher(ids, attention_mask=attention).logits[:, 1:-1].float().log_softmax(-1)
        sp, tq = p, q
        if top_k:
            support = p.topk(top_k, -1).indices
            sp, tq = p.gather(-1, support).log_softmax(-1), q.gather(-1, support).log_softmax(-1)
        kl = (sp.exp() * (sp - tq)).sum(-1)
    p_action = p.gather(-1, inputs["completion_ids"].unsqueeze(-1)).squeeze(-1)
    q_action = q.gather(-1, inputs["completion_ids"].unsqueeze(-1)).squeeze(-1)
    mask = inputs["completion_mask"] * inputs["tool_mask"]
    reference = ((p_action - q_action - kl).detach() * p_action * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, reference, atol=1e-6, rtol=1e-4)
    trainer.train()
    assert trainer.state.global_step == 1
    assert not torch.equal(student_before, student.transformer.h[0].attn.c_attn.weight)
    for name, parameter in teacher.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, teacher_before[name])


@pytest.mark.parametrize(
    "override",
    [
        {"beta": 0.5},
        {"top_p": 0.9},
        {"top_k": 20},
        {"min_p": 0.1},
        {"temperature": 0},
        {"repetition_penalty": 1.1},
        {"generation_kwargs": {"do_sample": False}},
        {"vopd_top_k": -1},
        {"vllm_structured_outputs_regex": "[0-9]+"},
    ],
)
def test_vopd_rejects_inconsistent_sampling(tmp_path, override):
    with pytest.raises(ValueError):
        DistillationConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, loss_type="vopd", **override)


def test_server_distillation_rejects_vopd(tmp_path):
    from trl.experimental.server_distillation import ServerDistillationConfig

    with pytest.raises(ValueError, match="local teacher"):
        ServerDistillationConfig(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            loss_type="vopd",
            teacher_model_server_url="http://localhost:8000",
            loss_top_k=1,
        )


@pytest.mark.parametrize("top_k", [0, 20])
def test_vopd_hub_model_train_save_reload(tmp_path, top_k):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    student = AutoModelForCausalLM.from_pretrained(model_id)
    teacher = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Distinct teacher parameters produce a nonzero training signal.
    with torch.no_grad():
        teacher.get_output_embeddings().weight.add_(0.01 * torch.randn_like(teacher.get_output_embeddings().weight))
    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        processing_class=tokenizer,
        args=DistillationConfig(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            report_to="none",
            loss_type="vopd",
            vopd_top_k=top_k,
            max_steps=1,
            per_device_train_batch_size=2,
            max_completion_length=3,
            save_strategy="no",
        ),
        train_dataset=Dataset.from_dict({"prompt": ["The sky is", "One plus one is"]}),
    )
    trainer.train()
    assert trainer.state.global_step == 1
    assert all(parameter.grad is None for parameter in teacher.parameters())
    trainer.save_model(str(tmp_path / "saved"))
    restored = AutoModelForCausalLM.from_pretrained(tmp_path / "saved")
    for name, parameter in student.named_parameters():
        torch.testing.assert_close(parameter.cpu(), dict(restored.named_parameters())[name])
