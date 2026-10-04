# Actual St4RTrack loss structure used by this comparison

The author-recommended `St4RTrack_Seqmode_reweightMax5.pth` has **two active
training terms**: confidence-weighted 3D tracking regression from head 1 and
confidence-weighted 3D reconstruction regression from head 2. A point cloud is
the representation used by both heads; it is not a third independent loss.
The joint attack adds the two active terms. The earlier globally median-aligned
tracking MSE is retained as a separately named evaluation-aligned surrogate.

The implemented five attacks decompose the active expression more finely:

| Objective ID | Maximized expression |
| --- | --- |
| `tracking_mse` | Previous evaluation-aligned tracking MSE, as the control |
| `tracking_3d` | `mean(c1_eff * e1)`, the weighted head-1 geometry term |
| `reconstruction_3d` | `mean(C2 * e2)`, the weighted head-2 geometry term |
| `confidence` | `-0.2 * (mean(log(c1_eff)) + mean(log(C2)))` |
| `joint_training` | The sum of the three expressions above, the complete active native loss |

Thus the standalone geometry attacks exclude the log-confidence penalty;
the joint attack recovers the full two-branch training objective. Confidence
still participates in weighted geometry gradients. A larger weighted loss can
reflect larger confidence rather than larger geometry error, so the study also
records unweighted normalized L21 errors, confidence means, and independently
aligned tracking/reconstruction APD and EPE for every attacked input. Maximizing
the negative log-confidence term pushes confidence downward. These are attacks
on inference objectives; they do not measure retraining after deleting a loss.

All source references below use official commit
`0f9a3f44a7ebac76600cd31ec9eea5228ad7db91`. Vendor files are unchanged.

## Checkpoint evidence and call graph

The local author checkpoint SHA256 is
`cae4712e0265f7d5ecadade19a0cb2ccf7b7e786fa0f230a80603a3437bcdfd3`.
Its actual `args.train_criterion` was read from the ZIP `data.pkl` metadata,
without loading model tensors or executing GPU inference:

```python
ConfLoss(Regr3D(L21, norm_mode='avg_dis', velo_loss=True),
         alpha=0.2, velo_weight=0, pose_weight=0, traj_weight=0.0,
         align3d_weight=0.0, depth_weight=0, cotracker=False,
         reweight_mode='max', reweight_scale=5.0)
```

This exactly agrees with the released
[sequence training script, lines 26–28](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/scripts_run/train_seq_reweight.sh#L26).
The checkpoint's model uses DPT pointmap heads, `output_mode='pts3d'`,
exponential depth parameterization and `conf_mode=('exp',1,inf)`.
The official README recommends this checkpoint by default:
[checkpoint instructions](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/README.md#download-checkpoints).

The training call graph is
`train_one_epoch → loss_of_one_batch → model(view1,view2) → criterion(gt1,gt2,pred1,pred2)`.
The criterion is instantiated from the actual training argument in
[training.py:315](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/training.py#L315),
and invoked in
[inference.py:168–173](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/inference.py#L168).
Head 1 returns `pts3d`; head 2 returns `pts3d_in_other_view`:
[model.py:302–310](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/model.py#L302).
Both coordinates refer to camera 0, while head 1 follows camera-0 content
through time and head 2 reconstructs the current frame's content.

The paper describes sparse mesh-vertex supervision for tracking and depth plus
camera supervision for reconstruction. It separately introduces trajectory,
depth and head-alignment terms for adaptation. See
[official paper Sections 3.2–3.3 and Appendix C](https://arxiv.org/html/2504.13152v1#S3.SS2)
and the [author project page](https://st4rtrack.github.io/).

## Exact active loss and coefficients

Let `P1[t,q]` be native raster query points and `P2[t,u,v]` the dense second
pointmap. Let `G1`, `G2` denote the fixed GT and `C1`, `C2` predicted confidence.
The pinned source scales both heads using the second head in each frame:

```text
s_pred[t] = max(sum_ALL_pixels(norm(P2[t])) / (3*H*W), 1e-8)
s_gt[t]   = max(sum_ALL_pixels(norm(G2[t])) / (3*H*W), 1e-8)
e1[t,q]   = norm(P1[t,q]/s_pred[t] - G1[t,q]/s_gt[t])
e2[t,u,v] = norm(P2[t,u,v]/s_pred[t] - G2[t,u,v]/s_gt[t])
L_track   = mean_native_queries_times(c1_eff*e1 - 0.2*log(max(c1_eff,1)))
L_recon   = mean_valid_GT_pixels_times(C2*e2 - 0.2*log(C2))
L_native  = L_track + L_recon
```

L21 is Euclidean distance, not squared distance. The effective dynamic-point
confidence is `5*max(predicted static C1 over the entire sequence).detach()`.
The static confidence retains its gradient. The dynamic membership is fixed
from GT: endpoint L1 movement divided by initial point norm, above the mean
of that score over the native query population. This differs from the
evaluation dynamic mask, which uses cumulative world movement above 0.01 m.
See [Regr3D:376–418](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/losses.py#L376)
and [ConfLoss:563–622](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/losses.py#L563).

The normalization above deliberately reproduces two upstream implementation
quirks. `Regr3D` passes masks positionally into `normalize_pointcloud_seq`, whose
extra `traj_mask` parameter leaves `valid2=None`. With no mask,
`invalid_to_zeros` counts XYZ components, `3*H*W`, rather than `H*W` points.
Thus **invalid depth pixels also affect normalization**, and the scale is a
third of the all-pixel mean norm. This affects normalized L21 and its gradient;
it is not silently corrected here. See
[losses.py:395–398](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/losses.py#L395),
[geometry.py:357–391](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/utils/geometry.py#L357),
and [misc.py:138–147](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/utils/misc.py#L138).

`reconstruction_norm` and `reconstruction_gt_norm` in the component API already
include the `/3` definition. Dense valid counts are reduced across all frames,
so reconstruction regression uses one global valid-pixel mean, not an equally
weighted mean of frame means. The static maximum likewise spans all frames.
Head 2 normalization remains in the autograd graph even for a head 1 tracking
attack. Confidence gradients remain active. Dynamic maximum weights have the
same stop-gradient as the author code and are recomputed at each PGD step.

### Interim full-frame DR example: normalization coupling

For `ds_mini/9c43b3-3_obj_source_left_3`, the saved 128-frame
[clean result](../runs/docker/lossstudy_allframes_20261004_183750/conditions/clean/ds_mini/9c43b3-3_obj_source_left_3/result.json)
and [tracking geometry attack result](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_3d/ds_mini/9c43b3-3_obj_source_left_3/result.json)
show a pronounced change in the head 2 normalization denominator. The vectors
are in [clean components](../runs/docker/lossstudy_allframes_20261004_183750/conditions/clean/ds_mini/9c43b3-3_obj_source_left_3/components.npz)
and [attacked components](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_3d/ds_mini/9c43b3-3_obj_source_left_3/components.npz);
the [fixed targets](../runs/docker/lossstudy_allframes_20261004_183750/sequences/ds_mini/9c43b3-3_obj_source_left_3/targets.npz)
contain the unchanged GT denominator.

| Denominator across 128 frames | Minimum | Median | Mean | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Clean predicted head 2 | 0.377407 | 0.380157 | 0.380431 | 0.383646 |
| Attacked predicted head 2 | 0.010932 | 0.021022 | 0.025249 | 0.303516 |
| Fixed GT head 2 | 1.037914 | 1.060702 | 1.056972 | 1.067059 |

Every predicted denominator decreased; none reached the `1e-8` clamp. The
same-frame attacked/clean ratio had median 0.055345 and mean 0.066344.
Unweighted native tracking L21 increased from 0.186681 to 136.851379, while
effective confidence decreased from 22.000959 to 4.022907 (raw confidence:
4.821521 to 1.000763). The weighted tracking attack objective was 696.459900.
Independently aligned benchmark tracking EPE changed from 0.251054 to
1.336160 m and APD from 72.244% to 21.033%; model-grid reconstruction EPE
changed from 0.150154 to 3.561525 m and APD from 84.237% to 4.842%.

Head 1 uses this head 2 denominator, so the observed contraction matters when
interpreting amplification of native normalized error. Normalized L21 and
benchmark metre EPE are different quantities and cannot be compared directly.
The denominator's isolated causal contribution was not tested: predictions
and confidence also changed. This is one interim DR case, not a completed
48-condition comparison or a ranking of the five objectives.

### Interim full-frame PO example

For `po_mini/dancingroom1_3rd2_4`, all three saved conditions cover 128 frames.
The ratio is the median of each frame's predicted head 2 denominator divided
by the same frame's clean denominator.

| Condition | Head 2 norm median | Same-frame norm ratio median | Native unweighted tracking L21 | Weighted native tracking | Tracking EPE (m) | Reconstruction EPE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Clean](../runs/docker/lossstudy_allframes_20261004_183750/conditions/clean/po_mini/dancingroom1_3rd2_4/result.json) | 0.318714 | 1.000000 | 0.799527 | 3.382432 | 0.373255 | 1.525110 |
| [MSE attack](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_mse/po_mini/dancingroom1_3rd2_4/result.json) | 0.166666 | 0.536510 | 9.016335 | 103.401108 | 14.861868 | 6.943280 |
| [Native tracking attack](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_3d/po_mini/dancingroom1_3rd2_4/result.json) | 0.009313 | 0.029084 | 364.804474 | 3024.171631 | 1.852127 | 9.076239 |

The denominator vectors are in [clean components](../runs/docker/lossstudy_allframes_20261004_183750/conditions/clean/po_mini/dancingroom1_3rd2_4/components.npz),
[MSE components](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_mse/po_mini/dancingroom1_3rd2_4/components.npz),
and [native tracking components](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_3d/po_mini/dancingroom1_3rd2_4/components.npz).
The [fixed targets](../runs/docker/lossstudy_allframes_20261004_183750/sequences/po_mini/dancingroom1_3rd2_4/targets.npz)
have GT norm median 1.518406 and range 1.281146–1.616276. The native attack's
predicted denominator decreased in 127/128 frames, with zero hits of the
`1e-8` clamp.

The native tracking attack has higher normalized tracking loss but smaller
benchmark tracking EPE in metres than the MSE attack; the denominator's
isolated contribution was not measured because predictions and confidence
also change. This is one interim PO example and establishes neither an
eight-clip/five-objective ranking nor full artifact verification.

## Inactive and adaptation-only terms

| Term | Actual Seq checkpoint weight | Released TTA script | Meaning |
|---|---:|---:|---|
| Head 1 confidence 3D tracking | 1 | Inactive on unsupervised `CustomDUSt3R` | Native active tracking component |
| Head 2 confidence 3D reconstruction | 1 | Inactive on unsupervised `CustomDUSt3R` | Native active reconstruction component |
| Velocity | 0 | Argument 1, but supervised branch inactive | Temporal normalized head 1 differences |
| Camera pose | 0 | 0 | Differentiable PnP rotation plus translation error |
| 2D trajectory | 0 | 0.5 | PnP-projected head 1 against CoTracker pseudo-labels |
| Scale-invariant depth | 0 | 10 | Predicted-camera head 2 Z against MoGe depth |
| 3D head consistency | 0 | 5 | Head 1 versus head 2 at CoTracker correspondences |
| Normal / ARAP | 0 defaults | 0 defaults | Optional source branches |

The released TTA configuration is
[train_tta.sh:22–24](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/scripts_run/train_tta.sh#L22).
`CustomDUSt3R` explicitly sets `supervised_label=0`:
[custom.py:195](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/datasets/custom.py#L195).
The source gates the two confidence losses and velocity on supervised data,
then adds auxiliary terms separately:
[losses.py:508–512](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/losses.py#L508)
and [losses.py:738–744](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/losses.py#L738).

Paper and implementation differ for TTA: Appendix C gives trajectory weight 1,
while the released script uses 0.5; Eq. 7 writes a squared trajectory loss,
while [losses.py:161](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/losses.py#L161)
computes RMSE. TTA also runs randomized CoTracker sampling and a randomized
PnP initializer, and `get_tracks_2d` silently substitutes identity camera poses
on solver exceptions. These terms are not active in this checkpoint's training
objective, and this experiment does not present GT substitutes for CoTracker
or MoGe as an exact replay of author TTA.

## Fixed WorldTrack GT and preprocessing contract

The inspected author PO and DR NPZ clips contain `images_jpeg_bytes`, camera
space `tracks_XYZ`, `visibility`, static `fx_fy_cx_cy`, dense `depth_map`, and
`extrinsics_w2c`. Dense reconstruction GT can therefore be formed exactly from
available depth and camera fields; **a sparse reconstruction surrogate is not
needed for these clips**. The loader fails if dense depth or camera fields are
missing. Sparse tracking GT is native in form: original training also used
sparse 4D query supervision, although WorldTrack's released benchmark query
population differs from the original training mesh vertices.

RGB and evaluation tracks use the unchanged official adapter:
`load_images(size=512,square_ok=True,crop=False)`, usually a final 512x288 grid.
Normalized cameras are `E_t @ inverse(E_0)`, with camera 0 as the world origin.
Dense GT uses nearest-neighbor depth resize to that exact model grid, scaled
x/y intrinsics, and camera-to-first-camera backprojection. This follows the
native depth-supervision form and preserves the evaluator RGB convention.
It does not reproduce training augmentation or the training camera resize's
half-pixel principal-point adjustment. Targets are real WorldTrack XYZ, not
depth inferred from attacked images or a changing pseudo-label model.
See [tapvid3d.py:458–475](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/datasets/tapvid3d.py#L458)
and [dense backprojection:537–554](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/datasets/tapvid3d.py#L537).

The fixed regression mask is positive finite depth with finite XYZ. Invalid
zero depth is not replaced with arbitrary zero world XYZ: after backprojection
it is the current camera origin in first-camera coordinates, matching the
official geometric transform. Such finite invalid values remain in the native
all-pixel normalization while excluded from regression. Nonfinite depth is
rejected because the literal native normalization would be nonfinite.

Native loss queries use round, clamp, and dense raster collision handling,
with the last original query retained at a colliding pixel and row-major
Boolean-index order. The training sources rasterize this way:
[PointOdyssey:347–354](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/datasets/pointodyssey.py#L347)
and [DynamicReplica:444–452](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/datasets/dynamic_replica.py#L444).
Official evaluation queries remain separately fixed at truncated coordinates;
later occlusion does not change either retained population.

Reconstruction APD/EPE computed directly on this model grid measures spatial
correspondence at the model resolution. It is **not** the official full
resolution reconstruction benchmark, which bilinearly upsamples predictions
to the original GT grid with `align_corners=False` before filtering:
[track_eval.py:214–224](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/track_eval.py#L214).

## API, provenance and validation

`load_component_sequence(...)` returns `(WorldTrackSequence, NativeTargets)`.
`NativeTargets` supports attribute and mapping access and `.to(device)`.
Its `reconstruction_gt`, `reconstruction_valid`, `native_tracks`, masks,
camera matrices and GT normalization scales remain fixed. Predicted head 2
normalization changes with the input and remains differentiable. Metadata stores exact
tensor/RGB/query hashes, native/evaluation query counts, collision count,
sampling rules, and hashes of official loss and geometry source files.

`NativeComponentForward(base,targets)` reuses the exact frozen/eval model from
`St4RTrackForward`. Each pair uses `(rgb[0],rgb[t])`, including the shared tensor
in both branches at `t=0`. `pair_fn` returns evaluation tracks plus native
head 1 queries/confidences and differentiable head 2 sufficient statistics.
`predict_with_maps` captures those statistics and full head 2 XYZ/confidence
on CPU, one pair at a time. Saving and reloading float RGB then recapturing
all tensors supports exact artifact replay.

`check_official_loss_parity` independently compares the full two-term loss and
its RGB gradient against the original official active loss definitions on a
bounded toy/real sequence. `check_official_component_oracle` performs the same
loss check and compares derivatives of head 1 XYZ/confidence and dense head 2
XYZ/confidence using captured CPU tensors, with no model or GPU invocation.

The full training loss import has optional camera/PyTorch3D dependencies.
Validation therefore compiles the **unchanged original AST nodes** for
`Sum`, `BaseCriterion`, `LLoss`, `L21Loss`, `Criterion`, `MultiLoss`, `Regr3D`,
and `ConfLoss`, and imports the actual official geometry helpers. All unused
auxiliary weights are zero, exactly as in the checkpoint. The report records
this extraction route, selected AST/source hashes, and that the full training
module was not imported. This oracle checks actual released source behavior,
including normalization quirks, rather than a second hand-written loss formula.

The CPU tests cover the normalization factor, invalid camera-origin pixels,
collision/query order, both-head and shared-anchor gradients, native source
value/gradient parity, missing dense GT rejection, and saved-float replay.

For real CUDA RGB-gradient validation only, `repeat_noise_check=True` performs
four backward passes on the same unchanged graph: native twice, then official
twice. The first three retain the graph; the last releases it. Loss values and
individual source component values still require strict tolerance checks.
The captured-map CPU oracle and the default RGB oracle retain strict
elementwise gradient checks; CPU callers cannot enable the CUDA noise mode.

The CUDA diagnostic records the original strict elementwise result, mismatch
count/fraction, maximum absolute error and nonzero-reference relative error.
It additionally measures both repeat differences and all four cross-formula
differences using float64 CPU L2 reductions. The relative denominator is
`max(norm(left),norm(right),1e-12)`. Both repeat noise and the worst cross
difference must remain below `1e-3` by default, and the worst cross difference
must also be no larger than `max(1e-5,1.5*max(measured repeat noise))`.
Consequently, a small systematic native/source difference cannot be excused
by unspecified GPU noise, and large repeat noise fails the absolute cap.
Failures carry the measured diagnostics in `NativeGradientParityError` and
its message, so a failed preflight remains reviewable. Enabling this mode does
not itself establish nondeterminism; the repeated-gradient measurements are
the required evidence.

## Optional runtime optimization, with the same loss and FP32 policy

`NativeComponentForward(base,context,optimize_runtime=True)` applies a reversible
fixed-grid optimization and preloads the fixed dense GT plus valid pixel
indices on the model device. The default is false. `forward.restore_runtime()`
restores the original model methods, disables this wrapper's optimized paths,
and releases its GT cache. Standalone
`apply_runtime_optimizations(model,image_hw=(H,W),vendor_root=...)` returns a
handle with `.restore()` and context-manager support for isolated benchmarks.

This optimization is restricted to the actual checkpoint's `VanillaDust3r`,
Python `RoPE2D`, and `PatchEmbedDust3R`. The encoder discards the temporal
position coordinate, and the Vanilla decoder also selects only 2D positions:
[model.py:190](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/model.py#L190)
and [model.py:271–276](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/model.py#L271).
Thus the changing synthetic time indices at larger batch sizes are not used
by this checkpoint; there is no source evidence that they alter its prediction
semantics. Batch changes can still change numerical kernel results and memory
usage and require separate real-model output/loss/gradient validation. This
implementation does not enable batched pairs or temporal variants.

The official Python RoPE reads `int(positions.max())` every call:
[pos_embed.py:148](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/croco/models/pos_embed.py#L148).
For a fixed 512x288 image with 16-pixel patches, its sequence length is already
known to be 32. The optimized instance uses that CPU constant and performs
the exact original cosine/sine lookup, embedding, rotation and concatenation
operations. It also corrects the positional cache key while preserving the
original Cartesian time/y/x coordinates and ordering. Vendor source and model
weights/state_dict remain unchanged. This is a runtime instance override with
explicit restore support and source hashes in metadata.

The wrapper supplies identical fixed H/W as a CPU shape tensor because the
official head wrapper reads `true_shape[0].cpu().tolist()`:
[misc.py:85–90](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/utils/misc.py#L85).
It caches fixed row-major valid pixel indices and uses index selection, avoiding
repeated CUDA Boolean-mask dynamic-shape indexing. Invalid points still
participate in normalization and remain excluded from regression, exactly as
before. All-valid frames use the equivalent contiguous flattened view.

`pair_frames_fn(frame0,framet,t)` additionally exposes a local RGB VJP without
building a full-video leaf gradient for every pair. At `t=0`, callers must pass
the identical Tensor leaf as both frame arguments, preserving both self-pair
paths. The existing video `pair_fn` remains available and uses that same
alias contract. The attack engine explicitly opts into the local replay API;
the loss values and sequence-global reductions remain unchanged.

The runtime metadata records each enabled optimization, fixed grid, original
source hashes and optimization source hash. CPU tests with real official small
model modules verify exact output/map/confidence and RoPE-gradient equality,
local/video gradient parity, native-source oracle parity, unchanged state_dict
and vendor source, valid cache keys, and restoration. Real-checkpoint GPU
throughput, memory and repeat-noise gradient measurements must establish the
benefit before the full experiment adopts the optimization.

## Current experiment frame policy

The current campaign uses every released frame, indices 0 through 127, for
each selected clip. Its eight matched clips contain four Point Odyssey and
four Dynamic Replica clips and preserve the original four measured videos.
All five objectives use the same complete 128-frame RGB and fixed GT.
The earlier 64-frame configurations and calibration records are historical
prefix runs and are not evidence that the complete-frame campaign finished.

The adapter reads the NPY headers of all five temporal fields and rejects
inconsistent raw lengths. Each sequence records original_frame_count,
source_frame_counts, all_frames_used and used_frame_indices. The current
configuration and manifest both declare all_frames=true and
campaign_expected_frames=128; the runner and final artifact audit enforce
those declarations against actual loaded and saved arrays.
