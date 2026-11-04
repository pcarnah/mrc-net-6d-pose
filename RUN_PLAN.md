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
  is `val` — webds test shards carry images only). The stereobj-1m mono-K1
  source is a dependency: all primary arms resume from its `epoch_002.pth`
  (the round recorded in §4b resumed from `epoch_001.pth` — its rows and the
  resid reference run are epoch_001-based; the uf_* ladder re-runs reg-only
  so the whole matrix shares the epoch_002 source).

## 4. Ablation plan v2 (unfrozen primary + controls — stereobj-1m)

**Revised after the first results round** (see §4b for what was measured and
why the protocol changed):

1. **Unfrozen is the primary protocol.** The unfrozen reg-only run (1 epoch
   from mono ep1) is the best row in the matched comparison of the
   epoch_001-source round; the frozen full
   arm is still below its init at a partial epoch. A frozen mono-trained
   trunk likely caps late fusion (no co-adaptation), and per unit of insight
   unfrozen is currently cheaper (best-in-table in one epoch vs below-init at
   a partial frozen epoch).
2. **`left_only`/`mean` resumed from the same mono checkpoint are the
   controls.** With an unfrozen trunk, "stereo beats mono" conflates (a) the
   stereo-eval BN operating point, (b) two-view training data, (c) the fusion
   pathway. `left_only` (no cross-view regression pathway) isolates (a)+(b);
   `mean` (fixed averaging, no learned fusion params) adds a parameter-free
   fusion baseline. Only the excess over these controls is attributable to
   fusion. `right_only` stays skipped.
3. **Frozen matrix demoted** to optional cheap secondary attribution — with
   the caveat that frozen results conflate freezing with pathway count (the
   completed frozen arm is full-fusion, the completed unfrozen arm reg-only).
4. **stereobj-1m is the ablation dataset; usprobe is the corollary** — train
   the winning config on usprobe from the usprobe mono checkpoint.

**Primary matrix (unfrozen, 1 epoch, K=1, all resumed from mono-K1
`epoch_002.pth`).**

| arm | fusion | clf | corr | status |
|---|---|---|---|---|
| `left_only` control | none | off | off | to run |
| `mean` control | fixed avg | off | off | to run |
| `uf_reg_k1` | cross_view residual | off | off | to run (a prior 1-epoch reg-only run exists: `chkpt_stereobj-1m_stereo_cross_K1_resid_reg_only_unfrozen` — keep as a reference point) |
| `uf_regclf_k1` | cross_view residual | on | off | to run |
| `uf_regcorr_k1` | cross_view residual | off | on | to run |
| `uf_full_k1` | cross_view residual | on | on | to run |

The reg-only → reg+clf → reg+corr → full ladder attributes each component
against the same controls; no corr-only arm (correlation needs a fusion conv
to enter) and no clf-only arm (no disparity cue). K=3 versions follow only
for arms that matter after the K=1 verdict.

**Common flags** (dev box, host-RAM bound — or the SLURM harness):

```
python main.py --dataset stereobj-1m --stereo_mode stereo \
    --pretrained chkpt_stereobj-1m_mono_K1/epoch_002.pth \
    --batch_size 4 --num_workers 0 --warmup_step 100 \
    --save_every_steps 500 --save_every_epochs 1 --no_compile \
    --chkpt_dir chkpt_stereobj-1m_uf_<arm>_k1 --log_dir logs_uf_<arm>_k1
```

Arm-specific: `left_only`/`mean` add `--fusion left_only|mean
--no-clf_fusion --no-fusion_corr`; `uf_reg_k1` adds `--no-clf_fusion
--no-fusion_corr`; `uf_regclf_k1` adds `--no-fusion_corr`; `uf_regcorr_k1`
adds `--no-clf_fusion`; `uf_full_k1` uses dataset defaults (clf+corr on).
No `--freeze_except_stereo` on primary arms.

On the cluster: stereobj-1m is ~24 h/epoch full-train — the primary matrix is
4 single-epoch jobs (`TIME` per job; `SAVE_EVERY_STEPS` checkpoints are the
recovery point — `main.py` has no resume).

**Eval discipline (matched comparison).**
- Always compare on the **common-frame set**: mono CSVs carry ~2 rows/frame
  (both eyes) vs stereo's 1, and 5000-instance caps land at different frames —
  of 715/716 frames only 358 are common. Comparing full CSVs mixes frame
  populations and produced the earlier false "stereo adds nothing" (the
  tube_rack −0.200 was largely this artifact). Filter every CSV to the common
  (scene_id, im_id) set before scoring; exclude centrifuge_tube; use the
  4-crop (multi) eval.
- Scored references: `reports/common_*_excl_ct.json` (add-s/proj, per-object).
- **z-error is the metric to watch**: ~96% of translation error is depth
  (z-share of euc ≈ 0.96–0.97 on matched frames), so add-s tracks |dz|
  directly. Track |dz| mean/median/p90 **and signed bias** per arm.
- **Step-0 stereo eval control** for every resumed run: the init checkpoint
  evaluated in stereo mode separates the BN operating-point shift from
  training effects (a mono-mode eval is not a bit-identical reference for a
  stereo model). Still pending for the frozen arm.
- Quick iteration: `scripts/eval_stereobj_subset.sh` (`CROP=multi`,
  `NUM_WORKERS` default 2).

**Priority order.**
1. The full primary matrix: controls (`left_only`, `mean`) + component ladder
   (`uf_reg_k1`, `uf_regclf_k1`, `uf_regcorr_k1`, `uf_full_k1`) — 6 runs,
   all resumed from mono ep2. The matched mono baseline for this round is a
   mono epoch_002 eval (plus its step-0 stereo control) — produce it before
   reading the ladder.
2. Step-0 stereo evals for resumed runs (cheap, interprets everything else).
3. K=3 versions of the winning arm(s).
4. Winner → usprobe corollary (`--dataset usprobe --stereo_mode stereo
   --pretrained <usprobe-mono-ep>`), eval via `inference_usprobe_stereo.py`.
5. Optional: frozen matrix for cheap per-component attribution.

Decision rule: read the ladder against the controls — reg-only > controls →
the learned fusion earns its keep; reg+clf/reg+corr > reg-only → that
component adds; full ≈ best single-component arm → no stacking benefit. Any
arm < its step-0 control → that pathway hurts at this training budget.

**Budget (rough).** Dev box at batch 4: unfrozen stereo K=1 ~1.5–2 h/epoch
(forward + full backward), K=3 ~2.5–3× K=1; cluster ~24 h/epoch full-train is
the upper bound. One epoch per arm; extend all arms identically if extended.

## 4b. Results so far (matched common-frame comparison, centrifuge_tube
excluded, multi crop, n=1790)

| run | add-s mean | median | AUC | acc.1d | proj mean | \|dz\| mean | dz bias |
|---|---|---|---|---|---|---|---|
| mono ep0 (source) | 12.60 | 10.72 | 0.450 | 0.823 | **19.62** | 11.93 | +1.42 |
| mono ep1 (init) | 12.54 | 9.36 | 0.483 | 0.844 | 31.29 | 12.13 | −2.69 |
| stereo ep0 full-ft | 16.84 | 13.58 | 0.361 | 0.711 | 34.22 | 16.58 | −3.56 |
| stereo ep1 full-ft | 12.65 | 9.65 | 0.478 | 0.846 | 19.01 | 12.21 | −4.39 |
| stereo ep2 full-ft | 11.77 | 9.01 | 0.500 | 0.863 | 21.97 | 11.41 | −1.00 |
| **stereo ep0 resid unfrozen** | **11.28** | **8.46** | **0.517** | **0.870** | 29.39 | **10.96** | **−0.28** |
| stereo ep0 scratch | 26.64 | 19.56 | 0.227 | 0.548 | 46.18 | 25.24 | +0.61 |
| chk_full frozen @120k | 13.24 | 10.20 | 0.454 | 0.791 | 26.98 | 12.84 | +2.80 |

Findings: stereo ep2 beats mono ep1 on every add-s metric and |dz|; the
1-epoch unfrozen reg-only run is best overall with near-zero depth bias;
scratch fails (mono init is load-bearing); the frozen full arm is below init
with a flipped depth bias (pending its step-0 control). tube_rack_50_ml's
apparent −0.200 regression was mostly the frame-population artifact; the
real residual is symmetry-branch churn (58/52 flips; the model selects the
180° twin rotation with identity translation — the twin's off-center
(101, −71, 0) mm mesh-frame offset makes that combination add-s-invalid).

**Gotchas.**
- **Eval batch composition changes results.** The backbone's BatchNorm is
  built with `track_running_stats=False`, which normalizes over the batch
  **even in eval** — predictions depend on what else is in the batch
  (measured: the same instance alone vs batched with another differs by
  ~0.08 in `quat_bin` logits and O(100 mm) in pose after the 4-crop argmax
  flips). The per-instance 4-crop batch is the calibrated eval unit; never
  batch instances together at inference, and note `--single_crop` (1-crop
  batches) also shifts the operating point. Throughput at eval comes from
  `--num_workers` prefetch, not from larger GPU batches.
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