# vOPD validation results

Review date: 2026-10-07. Commands below run from the development workspace root. The public snapshot uses the same relative source layout. The test environment is CPU-only; its versions are recorded in `test-environment.json` and `cpu-environment-freeze.txt`.

## Final regression runs

```bash
TMPDIR="$PWD/.tmp" HF_HOME="$PWD/.hf-cache" OMP_NUM_THREADS=2 \
.venv/bin/python -m pytest \
  repos/trl/tests/test_vopd.py \
  repos/trl/tests/test_distillation_trainer.py \
  repos/trl/tests/experimental/test_server_distillation_trainer.py \
  -k 'vopd or TestChunkedDivergenceLoss or test_config' -q --disable-warnings
# 69 passed, 73 deselected in 30.49s

TMPDIR="$PWD/.tmp" OMP_NUM_THREADS=2 PYTHONPATH=repos/verl \
.venv/bin/python -m pytest \
  repos/verl/tests/trainer/distillation/test_vopd_on_cpu.py \
  repos/verl/tests/workers/test_distillation_topk_symmetry_on_cpu.py \
  -q --disable-warnings
# 34 passed; see verl-audit-tests.log for elapsed time and warnings.
```

Logs: `trl-audit-tests.log` and `verl-audit-tests.log`. Earlier smaller runs are retained as historical evidence.

The TRL tests include exact action-expectation gradients, chunk/mask/temperature oracles, all-masked behavior, default JSD regressions, invalid sampler settings, the unsupported server-trainer guard, local tiny-model training, and two real Hub tiny-Qwen train/save/reload runs.

The verl tests include the production loss registry and PPO reduction, ragged alignment, teacher detachment, both FSDP output padding modes, Hydra recipe composition and invalid sampling settings, full-vocabulary wire ordering, separate sampled-token columns, full-V chunk boundaries, cross-trajectory request limits, and cancellation on request failure. The remote inference client is mocked; these tests do not start a vLLM server or a distributed FSDP engine.

## Official repository gates

Executed using the repositories' pinned `.pre-commit-config.yaml` versions, not merely the separately installed Ruff version:

```bash
# In each source checkout with the workspace environment active:
pre-commit install
pre-commit run --all-files
# In TRL also:
python scripts/add_copyrights.py
```

- TRL: Ruff 0.13.3 check/format, pinned doc-builder style, and the official copyright script pass.
- verl: Ruff 0.12.2 check/format, mypy 1.17, generated trainer configurations, documentation dates/docstrings, licenses, device/DataProto checks, test structure, naming, example names, uv policy, and compilation all pass.
- Logs: `trl-precommit-all.log`, `verl-precommit-all.log`, and `trl-copyright.log`.
- The initial scoped pre-commit logs contain formatter failures subsequently fixed; the all-files logs are the final result.
- New files were included in the git index before checks, so git-based copyright/license hooks did examine them.

## Examples and patches

```bash
.venv/bin/python repos/trl/examples/vopd/vopd.py --help
bash -n repos/verl/examples/on_policy_distillation_trainer/run_vopd_fsdp.sh
git -C repos/trl diff --check
git -C repos/verl diff --check
```

These checks pass. The patches include every changed and added file. Patch hashes and exact base commits are recorded in `patch-manifest.json`; reverse checks against the working trees and forward checks against clean base worktrees pass.

## Additional check that could not run

An attempted broader run including `tests/workers/test_megatron_distillation_only_on_cpu.py` failed during collection because `megatron` is not installed in this CPU environment. Its log is `verl-optional-megatron-collection.log`. The supported FSDP/vOPD suite was rerun separately and passed. No claim is made that the optional Megatron regression passed; vOPD explicitly rejects that backend.

## Remaining limits

No live vLLM/Ray GPU training, multi-rank optimizer run, GPU BF16/LoRA run, throughput measurement, or model-scale accuracy experiment was executed. These remain upstream submission gates. The full-V path is memory intensive; the selective-prefix top-k path needs realistic latency measurements. The shared existing verl empty-response/all-masked reductions also remain a runtime limitation to resolve or restrict before broader use.

vLLM 0.29 API support was inspected against its source, retained as `sampling_params.py.vllm029` and `outputs.py.vllm029`; source inspection does not replace a live integration run. The comprehensive audit and human/maintainer requirements are in `../REVIEW_CHECKLIST.md`.
