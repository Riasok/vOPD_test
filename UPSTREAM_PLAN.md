# Publishing and upstream contribution guide

## Current readiness

The audited work can be published to the requested public `Riasok/vOPD_test` repository. Upstream PR submission still needs the unchecked items in [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md), especially human ownership, maintainer agreement, live distributed validation, and experiments.

`vOPD_test` contains source snapshots, reference material, patches, and validation records. It is a standalone integration workspace, not a GitHub fork of either library. Submit the two changes through separate forks of their actual upstream repositories.

## 1. Agree on the contribution

Read the current guides again immediately before submission:

- [TRL contributor guide](https://github.com/huggingface/trl/blob/main/CONTRIBUTING.md), [agent instructions](https://github.com/huggingface/trl/blob/main/.ai/AGENTS.md), and [PR template](https://github.com/huggingface/trl/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
- [verl contributor guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md), [agent instructions](https://github.com/verl-project/verl/blob/main/AGENTS.md), and [PR template](https://github.com/verl-project/verl/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

Discuss the method, real use case, paper, example, and integration design in each project's issue tracker. **TRL requires first-time contributors to have an issue assigned before opening a PR, including a draft.** Its policy excludes fully AI-generated first-time PRs. The submitter must discuss eligibility honestly; do not relabel mostly generated code as human-written.

For verl, the human submitter must review every changed line, run relevant tests, and understand and defend the implementation. Include test commands/results, AI disclosure, attribution, and why the change does not duplicate existing work. Once the issue exists, run:

```bash
gh issue view ISSUE_NUMBER --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search 'ISSUE_NUMBER in:body'
gh pr list --repo verl-project/verl --state open --search 'vOPD'
gh pr list --repo verl-project/verl --state open --search 'distillation'
```

Replace `ISSUE_NUMBER` with the actual issue number. Search TRL too. Review related kernel and teacher PRs even when their names do not mention vOPD. The saved searches are dated evidence, not a permanent clearance.

## 2. Complete runtime and experimental validation

Use each project's supported dependencies; the recorded local CPU environment is not verl's pinned GPU environment.

1. TRL: real model train/save/reload already has CPU coverage. Add GPU BF16, LoRA, resume, and two-rank runs for full V and top-k.
2. verl: compare a live vLLM teacher's selective/full scores to a Transformers oracle; then run real Ray and FSDP/FSDP2 optimizer steps with correct masks and response alignment. Exercise distributed and empty-response behavior.
3. Compare baseline OPD, direct KL, vOPD-full, and vOPD-top-k with identical data/settings over several seeds. Record learning curves, evaluation, gradient variance/norm, KL, peak memory, throughput, and teacher request count/latency.
4. Use one update per rollout and one PPO epoch for the initial conditional-gradient comparison. Use matching vocabularies, untruncated sampling, unit temperature in verl, no clamps, and no task reward mixing for pure distillation.
5. Measure long-context prefix-request overhead before recommending the verl top-k path. Full V still transfers O(TV) data and is experimental.

Do not claim a paper speedup or distributed support based only on CPU unit tests.

## 3. Fork and push each upstream separately

Create `Riasok/trl` as a fork of `huggingface/trl`, and `Riasok/verl` as a fork of `verl-project/verl` using GitHub's Fork button. Authenticate Git locally through your usual credential helper; do not put a token in a remote URL or commit `.env`.

In the original development workspace, once the audited changes are committed on `feat/vopd`:

```bash
cd /data/minjaeoh/vopd-upstream/repos/trl
git remote add fork https://github.com/Riasok/trl.git
git fetch origin
git rebase origin/main
# Resolve any conflicts, review the resulting diff, and rerun checks.
make precommit
# Run the test commands in reports/VALIDATION.md and the completed GPU experiments.
git push -u fork feat/vopd

cd /data/minjaeoh/vopd-upstream/repos/verl
git remote add fork https://github.com/Riasok/verl.git
git fetch origin
git rebase origin/main
pre-commit install
pre-commit run --all-files --show-diff-on-failure
# Run the test commands in reports/VALIDATION.md and the completed GPU experiments.
git push -u fork feat/vopd
```

Activate that repository's validated environment before running its checks. If `fork` already exists, inspect it and use `git remote set-url fork ...` only if it is not your intended fork. The original clones' `origin` remotes point to upstream; push to `fork`.

If starting from a fresh clone of `vOPD_test`, its nested source snapshots are ordinary directories, not independent git repositories. Recreate proper upstream checkouts and apply the patches:

```bash
# Run in a directory containing the cloned vOPD_test folder.
git clone https://github.com/Riasok/trl.git trl-contribution
cd trl-contribution
git remote add upstream https://github.com/huggingface/trl.git
git fetch upstream
git switch -c feat/vopd adff4933bca014c6110436c8ba6bdbe88b28283c
git apply --check ../vOPD_test/patches/trl-vopd.patch
git apply ../vOPD_test/patches/trl-vopd.patch
git add --all
git commit -m 'Add vOPD control-variate distillation loss' -m 'Co-authored-by: Codex'
git rebase upstream/main
# Rerun checks and experiments before pushing.
git push -u origin feat/vopd
cd ..

git clone https://github.com/Riasok/verl.git verl-contribution
cd verl-contribution
git remote add upstream https://github.com/verl-project/verl.git
git fetch upstream
git switch -c feat/vopd 8718ca30a3f002f93b7c4fd99b9b2506718681bc
git apply --check ../vOPD_test/patches/verl-vopd.patch
git apply ../vOPD_test/patches/verl-vopd.patch
git add --all
git commit -m '[fsdp, algo, rollout] feat: add vOPD distillation' -m 'Co-authored-by: Codex'
git rebase upstream/main
# Rerun checks and experiments before pushing.
git push -u origin feat/vopd
```

Use your actual Git identity. Do not invent a sign-off or claim human testing that has not happened. Rebase can change the patch's behavior, so the archived test results do not validate a later rebase.

## 4. Open PRs when the gates are met

- TRL title: `Add vOPD control-variate loss to DistillationTrainer`.
- verl title: `[fsdp, algo, rollout] feat: add vOPD distillation`.
- Use the current official PR template, reference the agreed issue, include an API example and the estimator equation, and explain backend and cost limits.
- Attach exact test results and hardware/software versions, experiments/training curves/evaluation, and the current duplicate-search links.
- Disclose that these patches were mostly generated and revised with Codex; select TRL's actual AI-generated category while discussing its policy with maintainers.
- For verl, request CI through its `ci-request` Slack channel, or the Feishu alternative in the template, once ready. No message has been sent on your behalf.

No upstream issue, PR, or fork push is part of publishing the standalone `vOPD_test` workspace.
