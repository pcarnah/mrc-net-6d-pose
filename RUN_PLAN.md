# Run plan & working notes

Condensed history of the fork's work plus the going-forward ablation plan.
Last updated: 2026-11-04 (post `6f551f9`).

## 1. Where the model stands

`MRCNet` (Siamese two-stage: classification bootstrap → argmax render →
residual regression), with these stereo/usprobe additions:

- **Iterative refinement (Phase 1)** — weight-shared loop `K` times
  (`--n_refine_iters`); iteration 0 = original forward, iterations re-render
  the current estimate and compose incremental residuals. Deep supervision:
  reg losses every iteration, clf losses only at iteration 0. `K=1` is
  bit-for-bit the pre-refinement forward (`tools/regression_guard.py check`).
- **Reg fusion (`stereo_fuse`)** — left+right `reg_pool` (2048-ch, 16×16)
  concat **left first** → fusion conv → unchanged `RegressionHead` with left
  `fov`. `fusion_residual=True` = zero-init residual (`left + GN(conv(cat))`),
  so stereo-at-init == mono performance; default for new stereobj-1m/usprobe
  runs. Old checkpoints use `ConvGnReLU(4096, 2048)`.
- **Clf fusion (`stereo_fuse_clf`, `clf_fusion`)** — zero-init residual into
  the classifier: `clf_pool = left + GN(conv(cat(left, right)))` feeding the
  unchanged `ClassificationHead`. Right-eye real branch now computed before
  the classifier (order swap only). Bootstrap argmax is no longer strictly
  mono-left when enabled ⇒ K=1 stereo outputs change vs pre-clf-fusion
  checkpoints. Trained by the iteration-0 clf losses (GN gamma first).
- **Pool correlation (`stereo_corr`, `fusion_corr`)** — fresh
  `Correlation(patch_size=11, feature_dim=64)` probes over both eyes'
  `reg_pool`/`clf_pool`; 121-ch left→right volume concatenated into **both**
  fusion conv inputs (reg `4096+121→2048`, clf `2048+121→1024`). Disparity
  cue a 3×3 conv can't reach (±5 cells = ±80 px at 16×16). Recomputed per
  refinement iteration. Enters through the same zero-init GNs ⇒ inert at
  init; corr conv grads are gamma-scaled until the fusion GN gamma moves.
- **`--freeze_except_stereo`** — freezes everything outside
  `stereo_fuse.*`/`stereo_fuse_clf.*`/`stereo_corr.*`/`regressor.*`; frozen
  params excluded from the optimizer; flag recorded in `run_config`. For
  finetuning fusion+regressor on a pretrained checkpoint. DDP
  `static_graph=True` stays valid.
- All fusion flags live in `ModelConfig` (saved in checkpoints; inference
  rebuilds the right architecture); CLI overrides: `--fusion_residual`,
  `--clf_fusion`, `--fusion_corr` (BooleanOptionalAction, default = dataset
  config). `DATASET_CONFIG` sets `clf_fusion`/`fusion_corr` True for
  `stereobj-1m`/`usprobe` (inert while `stereo_fusion` is False).

## 2. History (condensed)

**Upstream base.** MRCNet CVPR 2024 release → custom usprobe dataset mods
(`b3afa99`) → `ycbv-ds` + `stereobj-1m` support (`90713cc`) → stereobj stereo
pathway (`ad21e8c`) → fusion ablation + mono/stereo overrides + SLURM harness
(`d26232e`).

**Modernization.** Deps modernized: mmcv/imgaug removed, NumPy 2, Windows
pytorch3d prebuilt wheels (`a76b947`); model/dataset decoupling
(`c9cab16`).

**Performance.** Mesh decimation (≤10k faces) + binned rasterizer
(`5bacdc3`); vectorized bbox map + cached render meshes (`695e1b0`);
`--bf16` autocast with fp32 renderer (`1a3e4cf`); 7× data pipeline: GPU
quat labels, post-crop aug, tar cache (`2a876cd`); stereobj → BOP-standard
millimetres + right-eye cx fix (`a0f533c`).

**Iterative refinement.** Loop + deep supervision (`ea9391a`); training/inference
plumbing (`7af0be5`, `19e5b0c`); `tools/check_refine.py` (`0e65b9a`); eager
fallback when torch.compile fails (`c9bda3b`); crop-batch helpers + refinement
stats in CSV (`c134c93`).

**Stereo fusion.** Residual formulation behind `--fusion_residual`
(`fe92640`); residual default for new stereobj-1m runs (`466f38e`).

**Usprobe stereo (2026-11-04 session).**
- `8677d9a` — `UsprobeDataset` scene pairing (2k/2k+1; baseline ~50.04 mm
  d435i / ~18.07 mm d405; `R_rel` SVD-orthonormalized), matched-scene
  validation (~78% of left instances survive; d405 train 000108–000143
  index-misaligned), stereo BOP inference (`inference_usprobe_stereo.py`),
  pairing validator (`tools/check_usprobe_stereo.py`), `get_metrics.py`.
- `b470826` — `--freeze_except_stereo`.
- `0acf20d` — clf fusion (`stereo_fuse_clf`), verified zero-init identity
  bit-for-bit + GN-gamma gradient on GPU.
- `6f551f9` — pool-level correlation (`stereo_corr`) into both fusion
  residuals; `check_stereo.py` extended (clf+corr identity vs fusion-free,
  corr grads finite, 251 params with grads).

**Gates.** Mono: `regression_guard.py check` + `check_refine.py` (bit-for-bit,
reference is env-sensitive — recapture after torch/CUDA upgrades). Stereo:
`tools/check_stereo.py` (passthrough wiring gate, K=1..3 finiteness,
clf+corr zero-init identity, backward reach, informational geometry, dataset
smoke). Usprobe pairing: `tools/check_usprobe_stereo.py`.

## 3. Runs in flight

- mono K=1 → 3 epochs; mono K=3 → 3 epochs.
- stereo K=1, **reg fusion only, full finetune**, 1 epoch, resumed from
  mono-K1 end-of-epoch-0. *Reference point, not part of the ablation matrix.*
- These runs are the **usprobe** lineage. The ablation matrix below runs on
  **stereobj-1m** (its `DATASET_CONFIG` defaults: `stereo_mode='stereo'`,
  `fusion_residual`/`clf_fusion`/`fusion_corr` all True; annotated eval split
  is `val` — webds test shards carry images only). stereobj-1m mono sources
  are a dependency: train mono K=1/K=3 (`--stereo_mode mono`) at least past
  the first save before the K-matched frozen arms can start.

## 4. Ablation plan (reg+clf+corr, K∈{1,3}, frozen — stereobj-1m)

**Principles.**
1. Frozen matrix = primary ablation: identical trunk, only fusion+regressor
   differ. Zero-init ⇒ every frozen run starts exactly at its mono source's
   performance.
2. Source checkpoint matched per K: K=1 arms ← stereobj-1m mono-K1 first
   checkpoint; K=3 arms ← stereobj-1m mono-K3 first checkpoint (matched trunk
   quality per K).
3. The running full-finetune stereo-K1 is a separate reference; rank
   frozen-vs-frozen, then full-finetune the winner.
4. **stereobj-1m is the ablation dataset; usprobe is the corollary** — after
   the winner is picked, train the winning config on usprobe (from the
   usprobe mono checkpoints) and report usprobe test as the transfer result.

**Matrix (8 frozen runs, `--dataset stereobj-1m --stereo_mode stereo`).**

| arm | `--clf_fusion` | `--fusion_corr` | K=1 | K=3 |
|---|---|---|---|---|
| reg-only | off | off | ✓ | ✓ |
| reg+clf | on | off | ✓ | ✓ |
| reg+corr | off | on | ✓ | ✓ |
| full | on | on | ✓ | ✓ |

No corr-only arm (correlation needs a fusion conv to enter); clf-only is
weak (no disparity cue), optional.

**Common flags** (dev box, host-RAM bound — or the SLURM harness for full
epochs):

```
python main.py --dataset stereobj-1m --stereo_mode stereo \
    --pretrained <stereobj-mono-ep001.pth> --freeze_except_stereo \
    --batch_size 4 --num_workers 0 --warmup_step 100 \
    --save_every_steps 500 --save_every_epochs 1 --no_compile \
    --chkpt_dir chk_stereobj_fr_<arm>_k<K> --log_dir logs_stereobj_fr_<arm>_k<K>
```

On the cluster (`scripts/slurm/submit_training.sh` → `train.sbatch`):
stereobj-1m is ~24 h/epoch full-train; frozen arms cut backward cost but the
forward (2× synthetic + right real branch, ×K iterations) still dominates —
budget accordingly (`TIME`, `N_EPOCHS`, `SAVE_EVERY_STEPS=2000`,
`SAVE_EVERY_EPOCHS=1` defaults; step checkpoints are the recovery point —
`main.py` has no resume).

Arm-specific: reg-only adds `--no-clf_fusion --no-fusion_corr`; reg+clf adds
`--no-fusion_corr`; reg+corr adds `--no-clf_fusion`; full uses defaults.

**Fusion-mode baselines (secondary axis, frozen K=1).** `--fusion` ablates
*how* the views are combined (`cross_view`/`mean`/`left_only`/`right_only`),
orthogonal to the clf/corr component axis above. Two cheap arms slot in
right after the reg-only frozen anchor, with `--no-clf_fusion
--no-fusion_corr` so they compare cleanly on the reg axis:

- `left_only` — the honest no-reg-fusion control (no cross-view regression
  pathway at all; the matrix's reg-only arm is a zero-init *learned*
  residual, which is not the same baseline).
- `mean` — fixed `0.5·(left+right)` averaging, no learned fusion params;
  tests whether the learned conv beats simple averaging.

`right_only` (regressor on the right eye alone) is low-value; skip. Both
arms bypass `stereo_fuse` entirely, so a frozen run trains only the
regressor; the reg-side correlation is inert under non-cross_view modes;
`left_only` still computes the right synthetic branch per iteration (known
wasted compute, ablation-only).

**Eval.** Per arm at its training K, on the stereobj-1m val split:

```
python inference_stereobj.py --dataset stereobj-1m \
    --checkpoint_name chk_stereobj_fr_<arm>_k<K> --model_name <file> \
    --output_suffix fr-<arm>_k<K> --n_refine_iters <K> --no_compile \
    --num_workers 4
```

`--num_workers N` prefetches the CPU prep (decode + Rz crop building) in
DataLoader workers so it overlaps the GPU call — the eval loop was
data-throughput bound; payloads are bit-identical to the serial path
(verified), differences beyond GPU run-to-run noise are not expected.

Quick iteration: `scripts/eval_stereobj_subset.sh` (subset eval; `CROP=multi`
re-enables the 4-crop argmax).

Baselines: stereobj-1m mono source checkpoints (one per K); optional step-0
sanity eval of any frozen run must reproduce mono numbers bit-for-bit
(zero-init guarantee; `check_stereo.py` gates the wiring).

**Priority order.**
1. full K=1, full K=3 — "does clf+corr help at all" vs reg-only.
2. reg+clf K=1, reg+corr K=1 — attribute the gain.
3. reg-only frozen K=1 — the frozen anchor.
4. `left_only` K=1, `mean` K=1 — fusion-mode baselines for the reg axis.
5. Remaining K=3 arms.
6. Winner → usprobe corollary: train the winning config on usprobe
   (`--dataset usprobe --stereo_mode stereo --pretrained <usprobe-mono-ep>`,
   same frozen protocol), eval usprobe test via `inference_usprobe_stereo.py`.

Decision rule: full ≫ reg-only → attribute, then full-finetune the winner
for the headline number. full ≈ reg-only → keep reg only.

**Budget (rough).** Dev box at batch 4: frozen stereo K=1 ~45–60 min/epoch
(forward unchanged, backward collapses to fusion+regressor), K=3 ~2.5–3× K=1
— but stereobj-1m shard counts dominate; on the cluster expect ~24 h/epoch
full-train as the upper bound. One epoch per arm is likely enough signal; if
extended, extend all arms identically.

**Gotchas.**
- `--warmup_step` must be < steps/epoch; check the stereobj-1m step count at
  the chosen batch size before picking a value.
- K=3 stereo OOM at batch 4 → `--refine_grad_ckpt` (safe without DDP).
- `--bf16` only if enabled for **all** arms (changes numerics).
- Checkpoints record `model_config` + `run_config`, so eval rebuilds each
  arm automatically; `inference_bop.py` rejects stereo checkpoints (correct).
- Frozen results are not directly comparable to the full-finetune stereo-K1
  run — different trainable sets.
- stereobj-1m mono runs need `--stereo_mode mono` (the dataset config
  defaults to stereo); flipping to mono restores the per-eye mode.