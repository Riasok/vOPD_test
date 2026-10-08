# vOPD — review only the changes

**Start with the two diffs below.** Each compares an unchanged upstream snapshot with one vOPD commit. Green/red lines show exactly what we added/changed; the surrounding framework code is context.

| Review | Complete working source | Total diff | Runtime/config diff |
|---|---|---|---|
| **[TRL changes](https://github.com/Riasok/vOPD_test/compare/trl-upstream...trl-vopd)** | [trl-vopd](https://github.com/Riasok/vOPD_test/tree/trl-vopd) | 7 files, +428/−2 | +91/−2 |
| **[verl changes](https://github.com/Riasok/vOPD_test/compare/verl-upstream...verl-vopd)** | [verl-vopd](https://github.com/Riasok/vOPD_test/tree/verl-vopd) | 16 files, +830/−40 | +290/−39 |

## Implementation in four steps

1. Add opt-in vOPD settings to the existing distillation trainers.
2. Compute `A = stop_gradient(log q(y) − log p(y) + KL_baseline)` and optimize the sampled student token. TRL uses `−A log p(y)`; verl uses its existing PPO surrogate.
3. Use either the full vocabulary or the **student’s top-k**, normalizing both distributions on that support. The sampled-token score always keeps full-vocabulary normalization.
4. Reuse existing batching, masking, checkpointed projection/PPO, and launchers; add regression tests and reject incompatible sampling/backend settings.

<details>
<summary><strong>Exact file-by-file change map</strong></summary>

Every filename below opens its **diff**, not the whole file. “Modified” means a small change inside an existing upstream file.

### TRL

| File | Change |
|---|---|
| [distillation_trainer.py](https://github.com/Riasok/vOPD_test/commit/98c95fa82cc7a01d332495b5e044921573863e95#diff-9743f20e0f16bf63047963b94c4caac21ce7fcb156b1adc41067f1e935f01aae) | Modified: sampled-token loss and detached full/top-k KL inside the existing chunked projection. |
| [distillation_config.py](https://github.com/Riasok/vOPD_test/commit/98c95fa82cc7a01d332495b5e044921573863e95#diff-b663b652750c79de3e11ae17ed2497c009fd5f7d1c8d5b0ab2476f8002fc1fe9) | Modified: `loss_type="vopd"`, `vopd_top_k`; sampler validation. |
| [server_distillation_config.py](https://github.com/Riasok/vOPD_test/commit/98c95fa82cc7a01d332495b5e044921573863e95#diff-698f86109d0c500494ac70fed4f6a0bad945affd1f79dbbd21292415ad7bfa99) | Modified: reject the unsupported remote-teacher variant. |
| [test_vopd.py](https://github.com/Riasok/vOPD_test/commit/98c95fa82cc7a01d332495b5e044921573863e95#diff-81743a94a31b5565a0d248f2db72e2df04691a59d81df0c8ea3234d3fd7880ce) | **New:** math, masks, config, real-model training and save/reload tests. |
| [vopd.py](https://github.com/Riasok/vOPD_test/commit/98c95fa82cc7a01d332495b5e044921573863e95#diff-301ef587a215933459faf1eee76da93493d558beb112f70858dd29eeb0771339) | **New:** thin wrapper around the existing distillation CLI. |
| [Trainer docs](https://github.com/Riasok/vOPD_test/commit/98c95fa82cc7a01d332495b5e044921573863e95#diff-3627d7ea9d494ee0001a18064dd53255aed938151295a4bb3af0c0e7cc23b8fe) · [paper index](https://github.com/Riasok/vOPD_test/commit/98c95fa82cc7a01d332495b5e044921573863e95#diff-124ec84eabfbcc029a0381c2db91ac268253f2afdbcf7103717723542bed073d) | Modified: usage and required paper entry. |

### verl

| File | Change |
|---|---|
| [FSDP `losses.py`](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-3f1997a94eb665f3b578b0c05dff0fbcf7ea1deaac67bdb4ad3544ed0f884dbf) | Modified: detached full/top-k baseline; full-V temporary buffers limited to 256-token chunks. |
| [Distillation `losses.py`](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-607dbfb52cc729a73439ee2e529d9514792fbdcfaae380b7ec1bf1acbc70513e) | Modified: register modes and feed the negative advantage into existing PPO. |
| [`teacher_manager.py`](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-de42ede357a1a0bc638c990dd19969ea9978f6d46e4921293ceb1d4d4b51847e) | Modified: score student-selected tokens plus a separate sampled token; bound and cancel requests. |
| [`agent_loop.py`](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-7d8baa13741a3ba9bfed072c1eb75619c83af4442d59598dca587e4fb49f9a3a) | Modified: carry student top-k IDs and prompt length to teacher scoring. |
| [Rollout `utils.py`](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-49ed2f7dbf7a4302f15ae448382b4a175af77927469455d7ec746015b3cddd0f) | Modified: move the existing prompt parser here; add full-V and selected-token parsing. |
| [vLLM `utils.py`](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-7d6f39c54c429c4dc4ce74b3dfebf0a3c37b7837a722293f3dc40a863050d60a) | Modified: re-export the moved parser; no duplicate implementation. |
| [`vllm_async_server.py`](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-1c9647b67401751906c41878feee82ddc4e050fbdefe47142df5b4c97653e8d7) | Modified: return requested token scores in request order. |
| [Validation](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-02f40d52bc58a9426b3bef3a228bb3afe78dcccebbf4cfb44fb3992bb4edd9e1) · [loss config](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-5c190103dd817d08b549ac60035cdb17eed62517bbf5894a99748afea5120108) · [YAML](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-03608345798b6a0ad9b954135495f892c418d3debd0899a7f2aff7800856c06d) | Modified: modes, sampling/backend guards, teacher settings. |
| [test_vopd_on_cpu.py](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-bb2e3180c568900fe40981d56a54f274a15b84af92cb6fb0238aea6551c1fe36) | **New:** estimator, transport, concurrency, chunking and config tests. |
| [FSDP output tests](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-9c35d80bd259f01fa859604fad403edbe341bba8fd3abb92e6c382e5581b92c9) · [license checker](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-4a8a460ae6eab46ea7658ab1e5d548702bf6b0907a0a797a097fea02a02e43be) | Modified: integration tests; accept the new contributor header’s 2026 year. |
| [run_vopd_fsdp.sh](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-49273a492ea13bcaad5887c0c164e40066c0b1572f17273921afcaca1634943c) | **New:** wrapper around the existing Qwen/FSDP launcher. |
| [OPD docs](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-5761976bc380f997615783b3873cb94f7ecbd74ae505c51a991081383be31861) · [example README](https://github.com/Riasok/vOPD_test/commit/e3685e26573b5f91215b93929774f717770558e2#diff-0d5b2cd4f3d20bcee639ebcd6d0d96ad9239f4078298e9d79a80f87fec0a72d0) | Modified: usage and limits. |

</details>

## Use the code

Clone the branch you need; it contains the full framework at its normal repository root. Follow that framework’s dependency setup, then the linked example.

```bash
git clone --single-branch -b trl-vopd https://github.com/Riasok/vOPD_test.git trl-vopd
git clone --single-branch -b verl-vopd https://github.com/Riasok/vOPD_test.git verl-vopd
```

TRL: `DistillationConfig(loss_type="vopd", vopd_top_k=20)`; use `0` for full V. verl: run `examples/on_policy_distillation_trainer/run_vopd_fsdp.sh`; set `VOPD_MODE=vopd_full` for full V.

## Status and provenance

- **Previously validated:** 69 TRL + 34 verl CPU tests and both complete pre-commit suites. This cleanup preserves the exact tested source; no training code was changed.
- **Still pending:** live GPU/Ray/vLLM runs and performance/accuracy experiments. verl top-k makes one teacher request per response token; full V transfers O(TV) data. Existing verl empty-response reductions need resolution before broader use.
- **Before upstream PRs:** human review/testing and honest AI disclosure; TRL first-time contributors need an assigned issue and must address its policy on AI-generated PRs. verl requires human ownership and algorithm experiments.
- **Pinned upstream snapshots, retrieved 2026-10-07:** [TRL `adff493`](https://github.com/huggingface/trl/commit/adff4933bca014c6110436c8ba6bdbe88b28283c) and [verl `8718ca3`](https://github.com/verl-project/verl/commit/8718ca30a3f002f93b7c4fd99b9b2506718681bc). The `*-upstream` branches have identical source trees; they do not auto-track upstream.
- [Paper](https://huggingface.co/papers/2605.07865) · [original method](https://github.com/holi-lab/vOPD) · [archived validation](https://github.com/Riasok/vOPD_test/blob/2b24723adcc82347d4059cc61b37e10df0b64627/reports/VALIDATION.md).

`main` now contains only this review map and `.gitignore`. Duplicated source snapshots, patches, logs, and long checklists were removed from the default view; the earlier workspace remains at tag `archive/initial-workspace`. Each framework branch retains its upstream license.
