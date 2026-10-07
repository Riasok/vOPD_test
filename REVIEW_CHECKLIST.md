# vOPD implementation review and submission checklist

Reviewed 2026-10-07. The code is suitable for publishing as an experimental integration workspace. **It is not ready to request upstream merge.** Distributed runtime validation, experiments, maintainer agreement, and the human submission requirements remain open.

## Instructions and official contribution requirements

The checkout instructions match the official upstream files on the review date; hashes and URLs are in `reports/official-guides-audit.json`.

| Repository | Authoritative instructions | Requirements affecting this work |
|---|---|---|
| TRL | [AGENTS.md](https://github.com/huggingface/trl/blob/main/.ai/AGENTS.md), [CONTRIBUTING.md](https://github.com/huggingface/trl/blob/main/CONTRIBUTING.md), [PR template](https://github.com/huggingface/trl/blob/main/.github/PULL_REQUEST_TEMPLATE.md) | Keep the trainer simple, preserve shared patterns, use real test objects, add the paper index entry, pass the prescribed checks. First-time contributors need an assigned issue before opening a PR. Fully AI-generated first-time PRs will not be reviewed. |
| verl | [AGENTS.md](https://github.com/verl-project/verl/blob/main/AGENTS.md), [CONTRIBUTING.md](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md), [PR template](https://github.com/verl-project/verl/blob/main/.github/PULL_REQUEST_TEMPLATE.md) | Use uv, check duplicates, follow title format, supply experiments for algorithms, run pre-commit and relevant tests, disclose AI assistance, include commit attribution. The human submitter must review every changed line, run tests, and understand the change. Request upstream CI through its designated channel when ready. |

## Mathematical implementation

For a fixed context, let `p` be the student distribution, `q` the teacher distribution, and `a ~ p` the sampled token. The minimized loss is

```text
A = stop_gradient(log q(a) - log p(a) + b)
L = -A * log p(a)
b_full = sum_v p(v) * (log p(v) - log q(v))
b_topk = sum_{v in S} p_S(v) * (log p_S(v) - log q_S(v))
S = student top-k, with p_S and q_S separately normalized on S
```

Because `b` is independent of the current sampled action and detached, its expected score-function gradient is zero. The full estimator's expected gradient matches the conditional reverse-KL gradient. This argument does not differentiate the distribution of preceding contexts and does not establish unbiasedness for arbitrary clipped or stale-policy PPO updates.

- [x] Correct sign: the baseline is added to the reward advantage and subtracted from the negative advantage.
- [x] Only the student sampled-token log probability carries the estimator gradient; teacher and baseline are detached.
- [x] Sampled probabilities retain full-vocabulary normalization, including samples outside top-k.
- [x] Both distributions are normalized on the same student support for the top-k baseline.
- [x] The sampled token occupies a separate teacher-score column and never changes the baseline support.
- [x] TRL exact action enumeration checks the expected gradient against direct reverse KL for full V, k=1, intermediate k, k=V, and k>V.
- [x] TRL temperature, chunk boundaries, completion alignment, tool masks, and fully masked batches have regression coverage.
- [x] verl tests exercise the production loss registry, PPO gradient at current policy equal to old policy, ragged packing, both FSDP output padding modes, and teacher detachment.
- [x] No claim that this baseline reduces variance on every context; it approximates the gradient-norm-weighted optimum.
- [ ] Verify the distributed/global normalization and numerical behavior on the supported GPU stack.
- [ ] Measure gradient variance, KL, training curves, and task accuracy over several seeds.

## Correctness fixes made during this review

- [x] Reject `vopd` in TRL's inherited `ServerDistillationConfig`; that trainer overrides the loss and otherwise silently accepts an unsupported objective.
- [x] Reject TRL constrained vLLM generation, which changes the distribution assumed by the score-function loss.
- [x] Reject greedy verl rollouts as well as nonunit temperature, truncation, and repetition penalties.
- [x] Preserve default JSD and existing distillation behavior through regression tests.
- [x] Bound selective teacher requests to eight per manager across simultaneous trajectories.
- [x] Use at most eight scoring tasks per trajectory and cancel/await outstanding tasks when a request fails or the trajectory is cancelled.
- [x] Validate support shape, distinct support IDs, prompt length, full-vocabulary ordering, and vocabulary size.
- [x] Enforce the FSDP/FSDP2 restriction in both configuration and the direct vOPD loss dispatch.
- [x] Add a real `trl-internal-testing` model/tokenizer train/save/reload test for both variants, supplementing the local tiny-model causal oracle.

## Modularity and efficiency

- [x] TRL extends its existing config, trainer, checkpointed chunk projection, and CLI; no extra trainer framework or redundant model-forward abstraction.
- [x] verl extends its loss registry and existing teacher/client interface; no second optimizer or PPO implementation.
- [x] Names distinguish student probabilities, teacher scores, sampled-token scores, and baseline support.
- [x] TRL reuses the full-vocabulary log-softmax for entropy when computing the full baseline.
- [x] verl computes the full baseline in 256-token chunks instead of materializing additional full-sequence FP32 softmax buffers.
- [x] Full-V no longer casts the full teacher-ID tensor to int64; that loss does not need teacher IDs.
- [x] Full teacher scores already have the vocabulary normalizer, so the full baseline does not normalize them again.
- [x] Full-V parsing uses linear vocabulary lookup, without sorting each token's vocabulary or rebuilding the ID list for every position.
- [ ] Benchmark peak memory and throughput. Chunked computation reduces temporary buffers; it does not eliminate the full teacher payload.
- [ ] Replace or optimize the verl selective-prefix workaround if long-context latency is unacceptable. Current vLLM 0.29 selected-output scoring costs one request per response token. Prefix payload work can grow quadratically with response length; score transfer is O(Tk).

## Repository checks and packaging

- [x] Official guides, agent instructions, PR templates, pinned hooks, and CI test discovery inspected.
- [x] All-files pre-commit passes in both repositories, including TRL doc-builder style and all verl local policy hooks.
- [x] TRL official copyright script passes.
- [x] Complete license headers added. verl's existing individual-contributor allow-list now includes 2026, matching the new test's actual year and contributor, without assigning it to another company.
- [x] TRL paper entry and trainer documentation updated; verl documentation timestamp, usage, and backend limits updated.
- [x] Both launch examples validated for CLI/shell syntax; example naming passes the official checker.
- [x] verl CPU CI discovers the added `*_on_cpu.py` tests under its existing Python path trigger. TRL uses its normal test directory. A new workflow is unnecessary for these CPU tests.
- [x] Current duplicate searches saved in `reports/duplicate-search-audit.json`. No `vOPD` or control-variate-distillation matches; related distillation PRs exist and need rechecking before submission.
- [x] Reviewable upstream patches and exact base revisions included.
- [ ] Once a design issue exists, read its comments and search open PRs for that issue number as verl requires.
- [ ] Refresh upstream and repeat checks before submission; related open kernel/teacher changes can conflict with these patches.

## Runtime and submission gates still open

- [ ] Human review of every changed line, rerun of relevant tests, and ownership of experimental claims.
- [ ] Maintainer agreement on the integration/API, including whether verl full-V belongs in the initial PR.
- [ ] TRL assigned issue before a first-time contributor PR. Disclose the actual mostly AI-generated origin; human review alone does not change that origin or automatically satisfy TRL's acceptance policy.
- [ ] Live vLLM selected-ID and full-prompt-score comparison against a Transformers teacher oracle.
- [ ] Real Ray rollout and FSDP/FSDP2 updates, including EOS, padding, response lengths, sequence parallelism, and distributed failure handling.
- [ ] GPU BF16 and multi-rank tests, checkpoint resume, and LoRA validation.
- [ ] Pure OPD versus direct KL versus vOPD-full versus vOPD-top-k experiments with identical prompts/seeds/batches; record accuracy, gradient statistics, peak memory, tokens/sec, and teacher request latency/count.
- [ ] Resolve or explicitly restrict empty-response/all-masked verl batches: its pre-existing common loss-range reduction calls min/max on an empty tensor, and a globally empty batch has a zero normalization denominator. TRL's all-masked path is tested; that result does not apply to verl.
- [ ] Confirm matching token IDs/tokenizers and compatible student/teacher vocabularies for actual models. Shape checks cannot prove semantic tokenizer identity.
- [ ] Request upstream CI when the PR is ready and supply algorithm experiments/training curves as required by verl.

The implementation is intentionally limited to the documented backends. Full-V teacher transport is O(TV), including scores and IDs, and can dominate memory. Neither implementation has a measured speedup from these tests. See `reports/VALIDATION.md` for exact executed commands and `UPSTREAM_PLAN.md` for the push/PR guide.
