Reproduction command: `bash scripts/ingestion/normalize/run.sh`

# D2 Normalization Report

- Live source root: `/K3D/Knowledge3D.local/galaxies`
- Normalized staging root: `/K3D/GitHub/Knowledge3D/scripts/ingestion/staging/D2_normalize/normalized`

## Pass Artifacts

- `normalized/*.jsonl`: 22 files
- `refs_rewrite_map.jsonl`: 276662 rows
- `bidirectional_edges.jsonl`: 329638 rows
- `orphan_targets.jsonl`: 14565304 rows
- `matryoshka_fills.jsonl`: 2731 rows
- `procedural_upgrades.jsonl`: 1995 rows

## D1 vs D2 Totals

| Metric | D1 | D2 | Delta |
| --- | ---: | ---: | ---: |
| Rows | 464334 | 296003 | -168331 |
| Missing IDs | 339 | 0 | -339 |
| Ad-hoc IDs | 160043 | 0 | -160043 |
| Duplicate Rows | 367275 | 82672 | -284603 |
| Missing Matryoshka | 70678 | 67947 | -2731 |
| Raw Payload | 1995 | 1995 | 0 |
| Unidirectional Sites | 18368305 | 144 | -18368161 |

## Per-Galaxy Delta

| Galaxy File | D1 Rows | D2 Rows | D1 Raw | D2 Raw | D1 Missing Matryoshka | D2 Missing Matryoshka | D1 Ad-hoc | D2 Ad-hoc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DObjects.jsonl | 1796 | 1796 | 0 | 0 | 0 | 0 | 1796 | 0 |
| Audio.jsonl | 3251 | 3251 | 0 | 0 | 0 | 0 | 3251 | 0 |
| Book_BiologyAtlas.jsonl | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 0 |
| Book_LanguageFoundations.jsonl | 17 | 17 | 0 | 0 | 0 | 0 | 17 | 0 |
| Book_MathematicsPrimer.jsonl | 17 | 17 | 0 | 0 | 0 | 0 | 17 | 0 |
| Book_PhysicsHandbook.jsonl | 18 | 18 | 0 | 0 | 0 | 0 | 18 | 0 |
| Book_ToolManual.jsonl | 17 | 17 | 0 | 0 | 0 | 0 | 17 | 0 |
| Character.jsonl | 2608 | 2608 | 1 | 1 | 2608 | 2608 | 456 | 0 |
| Drawing.jsonl | 1360 | 1360 | 214 | 214 | 0 | 0 | 1156 | 0 |
| Grammar.jsonl | 103039 | 51498 | 121 | 121 | 0 | 0 | 102914 | 0 |
| Language.jsonl | 116779 | 565 | 559 | 559 | 0 | 0 | 559 | 0 |
| Math.jsonl | 37732 | 37162 | 155 | 155 | 0 | 0 | 37711 | 0 |
| Meta.jsonl | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| Number.jsonl | 1001 | 1001 | 0 | 0 | 0 | 0 | 1001 | 0 |
| Reality.jsonl | 10874 | 10874 | 272 | 272 | 0 | 0 | 10872 | 0 |
| Tool.jsonl | 77 | 77 | 25 | 25 | 0 | 0 | 77 | 0 |
| Word.jsonl | 68070 | 68070 | 648 | 648 | 68070 | 65339 | 1 | 0 |
| game_mechanics.jsonl | 125 | 125 | 0 | 0 | 0 | 0 | 125 | 0 |
| meaning_layer_stars.jsonl | 117497 | 117491 | 0 | 0 | 0 | 0 | 0 | 0 |
| proceduralized_gsm8k_train_10.jsonl | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 0 |
| proceduralized_mmlu_val_10.jsonl | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 0 |
| reasoning_strategies.jsonl | 18 | 18 | 0 | 0 | 0 | 0 | 17 | 0 |

## Procedural Upgrade Outcome

- Upgraded rows: 0
- Deferred rows: 1995

## Hashes

- `scripts/ingestion/staging/D2_normalize/normalized/3DObjects.jsonl` `17d1e0dd6b433e0d4a70cc856152785e3a8bb0d2db9cccac7744033b67516453`
- `scripts/ingestion/staging/D2_normalize/normalized/Audio.jsonl` `f018457c5b14dcaeca298e00a92625dcd6ba727a0e12a54cb6cab0963d03041f`
- `scripts/ingestion/staging/D2_normalize/normalized/Book_BiologyAtlas.jsonl` `4e513b298702726e3bddc448f7664b7ee3132ff574fe6793fdba590baa42dcee`
- `scripts/ingestion/staging/D2_normalize/normalized/Book_LanguageFoundations.jsonl` `a74e231513fb92096384e806341688fbc49ee7dd8c8da7b8ef30605fcf66f047`
- `scripts/ingestion/staging/D2_normalize/normalized/Book_MathematicsPrimer.jsonl` `b84911515e929c69e12228e3272fa4ed1be2ab5ab4cbf38ad7b3ffbfb49d42cf`
- `scripts/ingestion/staging/D2_normalize/normalized/Book_PhysicsHandbook.jsonl` `64fe8162a8d6fc1e3e0d73faaee2b2733506718db3e818b9bde9a79401503a87`
- `scripts/ingestion/staging/D2_normalize/normalized/Book_ToolManual.jsonl` `2352aaa8c376a7c99c1de09f40f97c8ac35666ac67b5695738e2cd638abe89d2`
- `scripts/ingestion/staging/D2_normalize/normalized/Character.jsonl` `916134f73e0aa2d09cfe0dcca73214473536befc78172b2a6173bc48c50b49b5`
- `scripts/ingestion/staging/D2_normalize/normalized/Drawing.jsonl` `6467ae54508c71da82fc84007baf997dd63e4f0160410ff2e5f31241b5e0181e`
- `scripts/ingestion/staging/D2_normalize/normalized/Grammar.jsonl` `e446b8760aa5240b3339adbefc95bf7bf4fe39f9ecf221bedda77cc24e8d5688`
- `scripts/ingestion/staging/D2_normalize/normalized/Language.jsonl` `a8bbabf1b590f311d9a56efacd7f24c58f4df59dfca8e4907705c79869cf74c9`
- `scripts/ingestion/staging/D2_normalize/normalized/Math.jsonl` `e1b250a59a35a8542bab89d5f2c9de218ac7492589266de2036070f3d6d0a381`
- `scripts/ingestion/staging/D2_normalize/normalized/Meta.jsonl` `647fed76ecbb1677ab34b15c9d565009e40f1bcaf578435045a1d281653158f2`
- `scripts/ingestion/staging/D2_normalize/normalized/Number.jsonl` `86e0a9f217b11afe9f9559a04328db65ed1d69960e440576229d53fccfb4f6f8`
- `scripts/ingestion/staging/D2_normalize/normalized/Reality.jsonl` `f79b79be4056302dea63ea3538ba87145e2e42863e17646a007da8fd8d704286`
- `scripts/ingestion/staging/D2_normalize/normalized/Tool.jsonl` `51fbfa9c0caf5965db08dbf256362da69c53d053a39bfeb07516ffb8ce68fb24`
- `scripts/ingestion/staging/D2_normalize/normalized/Word.jsonl` `ac26b5dc48ddb759300f7f1c9801b96f4640a5e571ede2f3589f7c9c494d6e8f`
- `scripts/ingestion/staging/D2_normalize/normalized/game_mechanics.jsonl` `6132a87aaf3f0d2adf2500b77e7f09830b36a63b3cdd0f2ebcc564e3539bfb99`
- `scripts/ingestion/staging/D2_normalize/normalized/meaning_layer_stars.jsonl` `e28baaa4ddb7bbb79ff601049b5dcbad1d7a71a091882567c2984f1d4f9dc6aa`
- `scripts/ingestion/staging/D2_normalize/normalized/proceduralized_gsm8k_train_10.jsonl` `d6d4da6c0bfa81310d7f9fea7b4edb7395e2ade8333ac4f038ffc83b7a95278d`
- `scripts/ingestion/staging/D2_normalize/normalized/proceduralized_mmlu_val_10.jsonl` `bbcf5c3d912a1380af71ffdad1d24103296a3fc42f0e24dc2bc61448c3327c7d`
- `scripts/ingestion/staging/D2_normalize/normalized/reasoning_strategies.jsonl` `ad715e00fd308e8f679c9c304178ed3bce466af5ac10b49e0979fd26288109f4`
- `scripts/ingestion/staging/D2_normalize/refs_rewrite_map.jsonl` `53d7d2eb6061509f793b51d71ea033fc5e50cec17772832f48f2dbf174a32206`
- `scripts/ingestion/staging/D2_normalize/bidirectional_edges.jsonl` `f374ef4485edccbbb3163bf33fb0e98be5541d049923563a140d61ac4f496629`
- `scripts/ingestion/staging/D2_normalize/orphan_targets.jsonl` `4c3785b5fe07650f95ff44ecf73ae21b77edc0610f4d8c40f0dd7af01324e2a5`
- `scripts/ingestion/staging/D2_normalize/matryoshka_fills.jsonl` `609695245ac606c38d439652ec11712e6dfec9075ff95ac76b61fb93204c351f`
- `scripts/ingestion/staging/D2_normalize/procedural_upgrades.jsonl` `975d2b7d7b21f2873816f8c6d6d10bc668feaf5cad492ff3f1a644bbcc50d607`
- `scripts/ingestion/staging/D2_normalize/re_audit/galaxy_census.jsonl` `17a6f3498f6aba1b1f14492cff9734d6cea14b619b99f6f0c2e31cdc61a7df8f`
- `scripts/ingestion/staging/D2_normalize/re_audit/violations.jsonl` `9dcecffd87695f3224ba05c042eaadc462d68bef06b9a4a039901a4b37988c15`
