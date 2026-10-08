# vOPD

TRL: full-vocabulary or student top-k baseline. **verl: top-k only.** Both extend the existing distillation flow. Review links show only the final patch against upstream.

| Review only the changes | Complete codebase | Total diff | Runtime/config diff |
|---|---|---|---|
| **[TRL changes](https://github.com/Riasok/vOPD_test/commit/9d2394c3a11cc9aaf76b4eed06a6f58a57eab83e)** | [trl-vopd](https://github.com/Riasok/vOPD_test/tree/trl-vopd) | 6 files, +299/−2 | +83/−2 |
| **[verl changes](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f)** | [verl-vopd](https://github.com/Riasok/vOPD_test/tree/verl-vopd) | 9 files, +497/−2 | +173/−0 |

## What to review

1. **TRL loss:** [chunked estimator](https://github.com/Riasok/vOPD_test/commit/9d2394c3a11cc9aaf76b4eed06a6f58a57eab83e#diff-9743f20e0f16bf63047963b94c4caac21ce7fcb156b1adc41067f1e935f01aae) — `A = stop_gradient(log q(y) − log p(y) + KL)`; minimize `−A log p(y)`.
2. **verl loss:** [top-k baseline](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f#diff-3f1997a94eb665f3b578b0c05dff0fbcf7ea1deaac67bdb4ad3544ed0f884dbf) → [existing PPO loss](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f#diff-607dbfb52cc729a73439ee2e529d9514792fbdcfaae380b7ec1bf1acbc70513e) — normalize both distributions on student top-k; keep the sampled-token score separate.
3. **Teacher plumbing:** [rollout IDs](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f#diff-7d8baa13741a3ba9bfed072c1eb75619c83af4442d59598dca587e4fb49f9a3a) → [prefix scoring](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f#diff-de42ede357a1a0bc638c990dd19969ea9978f6d46e4921293ceb1d4d4b51847e) → [vLLM scores](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f#diff-1c9647b67401751906c41878feee82ddc4e050fbdefe47142df5b4c97653e8d7) — token `t` uses prediction row `t−1`; requests are bounded and cancelled on failure.
4. **Guards:** [TRL settings](https://github.com/Riasok/vOPD_test/commit/9d2394c3a11cc9aaf76b4eed06a6f58a57eab83e#diff-b663b652750c79de3e11ae17ed2497c009fd5f7d1c8d5b0ab2476f8002fc1fe9) / [local teacher only](https://github.com/Riasok/vOPD_test/commit/9d2394c3a11cc9aaf76b4eed06a6f58a57eab83e#diff-698f86109d0c500494ac70fed4f6a0bad945affd1f79dbbd21292415ad7bfa99); [verl backend/sampling](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f#diff-02f40d52bc58a9426b3bef3a228bb3afe78dcccebbf4cfb44fb3992bb4edd9e1) / [teacher/loss settings](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f#diff-5c190103dd817d08b549ac60035cdb17eed62517bbf5894a99748afea5120108).
5. **Regressions:** [new TRL tests](https://github.com/Riasok/vOPD_test/commit/9d2394c3a11cc9aaf76b4eed06a6f58a57eab83e#diff-81743a94a31b5565a0d248f2db72e2df04691a59d81df0c8ea3234d3fd7880ce) and [extended verl tests](https://github.com/Riasok/vOPD_test/commit/efde882bb9deb7cff24f1e260a062c0d7678375f#diff-9c35d80bd259f01fa859604fad403edbe341bba8fd3abb92e6c382e5581b92c9) — gradient, masks, support separation, teacher isolation, and request handling. Remaining diff: usage docs and TRL’s required paper entry.

## Use

```bash
git clone --single-branch -b trl-vopd https://github.com/Riasok/vOPD_test.git trl-vopd
git clone --single-branch -b verl-vopd https://github.com/Riasok/vOPD_test.git verl-vopd
```

- **TRL:** existing `trl distillation` command with `--loss_type vopd --vopd_top_k 20` (`0` = full vocabulary). [Usage](https://github.com/Riasok/vOPD_test/blob/trl-vopd/docs/source/distillation_trainer.md#vopd-a-detached-control-variate-baseline).
- **verl:** existing Qwen/FSDP launcher with `loss_mode=vopd_topk`. [Exact overrides](https://github.com/Riasok/vOPD_test/blob/verl-vopd/docs/algo/opd.md#vopd-distillation-with-a-control-variate-baseline).

**Status:** static checks pass; tests were not rerun for this revision. Live GPU/vLLM validation remains pending. verl requires text-only, single-turn FSDP/FSDP2 + vLLM and makes one teacher prefix request per response token.

No custom launcher wrappers or separate parser layer. `main` contains this README and `.gitignore`; framework branches contain the complete source, tests, docs, and upstream licenses.

[Paper](https://huggingface.co/papers/2605.07865) · Pinned baselines: [TRL `adff493`](https://github.com/huggingface/trl/commit/adff4933bca014c6110436c8ba6bdbe88b28283c) / [verl `8718ca3`](https://github.com/verl-project/verl/commit/8718ca30a3f002f93b7c4fd99b9b2506718681bc).
