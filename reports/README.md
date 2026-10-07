# Validation record index

Use `VALIDATION.md`, `trl-audit-tests.log`, `verl-audit-tests.log`, and the `*-precommit-all.log` files for the final reviewed result. `official-guides-audit.json` records source hashes; `duplicate-search-audit.json` records the current complete search results; `patch-manifest.json` records the published patch contents.

The earlier `completion-audit.json`, `upstream-heads.json`, `*-regression.log`, `*-vopd-tests.log`, and original `upstream-search.json` describe the first implementation pass. They are retained as history and do not supersede the final review. Initial scoped pre-commit logs show formatting issues that the final all-files pass resolved.

`verl-optional-megatron-collection.log` is an unsuccessful optional regression attempt caused by a missing dependency. It is not part of the successful test count.
