# vOPD test and upstream integration workspace

Created 2026-10-07. Public destination: [Riasok/vOPD_test](https://github.com/Riasok/vOPD_test).

**Experimental integration; upstream submission gates remain open.** The source snapshots, patches, tests,
and audit records are included. Credentials, Python environments, caches, and downloaded model weights are excluded.

## Sources

- Paper: [KL for a KL: On-Policy Distillation with Control Variate Baseline](https://arxiv.org/abs/2605.07865).
  [Paper reference](paper/README.md).
- Original implementation: [repos/vOPD](repos/vOPD), cloned from `holi-lab/vOPD`.
  Its `src/opd_trainer.py` supplies the reference conventions; the new integrations implement the equations in each
  framework's current infrastructure. The pre-existing development checkout was left untouched.
- Current upstream clones: [repos/trl](repos/trl), base `adff4933bca014c6110436c8ba6bdbe88b28283c`, and
  [repos/verl](repos/verl), base `8718ca30a3f002f93b7c4fd99b9b2506718681bc`.
  The original development checkouts use `feat/vopd`; this public workspace preserves their source snapshots.
  [patches](patches) contains the changes against the base revisions.
- [reports/sources.json](reports/sources.json) records exact revisions and retrieval time.

## Implementations

| Framework | Variant | Entry point | Main code |
|---|---|---|---|
| TRL | Full V | `DistillationConfig(loss_type="vopd", vopd_top_k=0)` | `repos/trl/trl/trainer/distillation_trainer.py` |
| TRL | Student top-k | same, with `vopd_top_k=20` | same |
| verl | Student rollout top-k | `distillation.distillation_loss.loss_mode=vopd_topk` | `repos/verl/verl/trainer/distillation/{losses.py,fsdp/losses.py}` |
| verl | Full V, experimental | same, with `loss_mode=vopd_full` | same |

TRL reuses the existing chunked LM-head projection and local teacher. It computes
`A = stop_gradient(log q(y) - log p(y) + KL_baseline)` and minimizes `-A * log p(y)`.
Only the sampled-token log probability carries gradients. For top-k, the baseline
uses the student's k most likely tokens and renormalizes both distributions there.
The sampled token retains its full-distribution probability, even outside that support.

verl feeds the same advantage into its existing PPO distillation loss. With current
policy equal to rollout policy, its gradient equals the score-function loss above.
Top-k support comes from the student's rollout; the actor recomputes the baseline
probabilities on that fixed support. Teacher scores are fetched by
`verl/experimental/teacher_loop/teacher_manager.py`, with vLLM adapter support in
`verl/workers/rollout/vllm_rollout/vllm_async_server.py` and parsing in
`verl/workers/rollout/utils.py`.

### Run examples

From `repos/trl`, in a TRL GPU environment:

```bash
accelerate launch examples/vopd/vopd.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --teacher_model_name_or_path Qwen/Qwen3-4B \
  --dataset_name trl-lib/ultrafeedback-prompt \
  --loss_type vopd --vopd_top_k 20 \
  --output_dir outputs/vopd --max_completion_length 256
```

Set `--vopd_top_k 0` for full V. These are usage examples, not completed model-scale benchmark runs.

From `repos/verl`, after preparing the datasets and resource settings of its existing OPD example:

```bash
VOPD_TOPK=20 bash examples/on_policy_distillation_trainer/run_vopd_fsdp.sh
VOPD_MODE=vopd_full TRAIN_BATCH_SIZE=2 PPO_MINI_BATCH_SIZE=2 \
MAX_PROMPT_LENGTH=32 MAX_RESPONSE_LENGTH=32 \
bash examples/on_policy_distillation_trainer/run_vopd_fsdp.sh
```

The wrapper accepts the model/resource environment variables and Hydra overrides
used by `run_qwen3_8b_fsdp.sh`. It is not a zero-configuration launch on this shared machine.

## Scope and limits

- Both implementations require matching tokenizer/vocabulary IDs, untruncated student sampling, and a detached baseline.
- TRL supports its existing local teacher and generation backends; its tests include real tiny-model training on CPU.
- verl supports FSDP/FSDP2, text-only single-turn trajectories, temperature 1, and a vLLM teacher.
  Megatron, fused kernels, and static packed padding are rejected.
- verl's pinned vLLM 0.29 supports selected **output** token scores. Top-k therefore makes one teacher prefix request per
  response token, with up to eight requests in flight per teacher manager and O(Tk) score transfer. It requires `1 <= k <= 127`.
  This is exact scoring on the selected student support, but throughput needs measurement. It does not reproduce the
  paper's same-forward-pass cost advantage across separate inference/training services.
- Full V is technically possible with vLLM `prompt_logprobs=-1`; this patch adds the missing extraction and loss plumbing.
  It is expensive: O(TV) scores and IDs across the teacher/actor boundary, about 2.3 GiB for T=2048 and V=150000 before
  copies. Use short contexts/small batches. The top-k transport avoids this full-vocabulary transfer.
- PPO reuse/clipping and sampler/trainer differences retain their normal effects. The baseline itself is action-independent;
  this does not make an arbitrary off-policy or clipped training recipe unbiased.
- Live vLLM/Ray GPU training and model-scale accuracy/throughput benchmarks have not been run. CPU results prove the
  tested math, masking, loss routing, and transport contracts, not distributed runtime or benchmark performance.

Validation results and exact counts are recorded in [reports/VALIDATION.md](reports/VALIDATION.md). See [reports/VALIDATION.md](reports/VALIDATION.md)
for exact commands, coverage, and limitations. Complete review patches are in [patches](patches).

See [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) for the comprehensive audit and remaining gates,
and [UPSTREAM_PLAN.md](UPSTREAM_PLAN.md) for submission steps and [reports](reports) for validation artifacts.
