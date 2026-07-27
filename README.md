# HaMeR confidence

**Can we predict when a hand-pose model is wrong?** 
This is a per-frame confidence estimation for HaMeR under hand-object occlusion, 
where we'd rather drop a bad frame than imitate it.

*Free consistency signals rank unreliable frames nearly as well as an oracle
with access to the true error and the trained confidence head does not.*

## The question

Hand-pose reconstruction (HaMeR) fails most under hand-object occlusion,
exactly the frames that matter in manipulation videos. Demonstration
pipelines currently treat every reconstructed trajectory as equally
trustworthy. This project asks: **is there a per-frame score that knows
when the reconstruction is lying**, cheap enough to run inside a data
pipeline, and good enough to filter on?

## Method

Base model: [HaMeR](https://github.com/geopavlakos/hamer) (frozen).
Four per-frame uncertainty estimators, from expensive to free:

| Estimator | Idea | Cost |
|---|---|---|
| TTA ensemble | variance across jittered-bbox re-runs | n× inference |
| Temporal jitter | inconsistency vs. local median trajectory | **free** |
| Reprojection | disagreement with an independent 2D detector (MediaPipe) | 1 light model |
| Learned head | MLP on frozen ViT features → predicted error | small train |

Evaluated by score↔error correlation, sparsification (AUSE), failure
detection (AUROC @ 20 mm MPJPE), and a demonstration-filtering experiment,
all stratified by occlusion severity.

## Takeaways:

1. **Cheap beats clever.** The two near-free signals track the oracle's
   sparsification curve almost exactly; the trained head finishes last.
2. **Filtering works:** keeping the most-confident half of frames cuts mean
   error 44% (30.1 → 16.9 mm).
3. **The signal is strongest where it matters:** correlation peaks in the
   worst occlusion bin (ρ = 0.81).