# HaMeR confidence

There is quite assumption inside every pipeline that learns robot manipulation from human video: that the reconstrution is true. A camera watches a person pick up a mug. A pose model, let's say HaMeR, turns each frame into 21 3-dimensional joint positions. Those trajectories become demonstrations, and a policy learns to imitate them. At no point does anything in that chain ask whether the hand it is imitating was ever really there.

Most of the time this assumption is harmless, because most of the time the reconstruction is roughly right. The trouble is that the failures are very likely not randomly distributed. They could cluster precisely where the fingers wrap around the object, where the hand is occluded by the very thing it is manipulating. Which is to say, the model fails hardest exactly during the moments that carry the manipulation information. A pipeline that treats all frames equally is not just averaging in some noise; it is systematically weighting its worst data at the moments that matter most.

We ask whether that can be fixed cheaply, not by making reconstruction better because HaMeR stays frozen throughout, but by computing. Thus, the question becomes: can we compute a number, per frame, that perdicts whether the reconstrution is wrong without access to ground truth, while being cheap enough to run over a whole dataset? If such a number exists and is cheap enough to run across an entire dataset, the pipeline can stop imitating everything and start imitating only what is credible. 

### The Vision: Consistency as a proxy for correctness

The obvious obstacle is that we cannot check correctness without ground truth, and ground truth is exactly what a video-scraping pipeline lacks. However, there is a way around this, and it is the conceptual core of this project. We can check consistency, and inconsistency is evidence of error.

**Three kinds of consistency**

There is consistency across perturbations: run the model again on a slightly jittered crop of the same frame. A confident model gives nearly the same answer; a model that is guessing swings wildly. This is the intuition behind ensembles and MC-dropout, and it has a Bayesian reading. Dropout samples approximate draws from a posterior, so their spread approximates epistemic uncertainity. Formally, for $M$ perturbed runs,

$$u(t) = \frac{1}{21}\sum_j \|\mathrm{std}_m\,\hat p_j^{(m)}(t)\|_2$$

The cost is $M$ forward passes, which is the price we pay for the most direct measure of the model's own uncertainty.

There is consistency across time. Real hands move smoothly and do not teleport. A reconstruction that jumps between adjacent frames is announcing that at least one of those frames is wrong. The natural way to measure this is against a locally median-filtered trajectory,

$$\tilde p_j(t) = \underset{|t'-t|\le w/2}{\mathrm{median}}\ \hat p_j(t'), \qquad
u(t) = \frac{1}{21}\sum_j \left\|\hat p_j(t) - \tilde p_j(t)\right\|_2$$

and the choice of median rather than velocity is where the design earns its keep. Raw frame-to-frame velocity punishes genuinely fast motion, which is not an error because hands do move fast. Distance to a local median punishes only departures from the trajectory, and because the median is robust, a single bad frame does not corrupt the reference againts which it is judged. This estimator costs nothing at all, and it is arithmetic on outputs we already have.

And there is consistency across models. Project the 3D prediction back into the image and compare it against an independent 2D detector:

$$\pi(p) = \frac{(Kp)_{xy}}{(Kp)_z}, \qquad
u(t) = \sum_j w_j \left\|\pi(\hat p_j(t)) - d_j(t)\right\|_2$$

The leverage here is that MediaPipe is not a smaller HaMeR. It is a different family with different failure modes, which means its errors are largely decorrelated. Two systems that fail differently agreeing on an answer is far stronger evidence than either system's own confidence.

Against these three the project sets a fourth, more fashionable approach: a small MLP on frozen backbone features, trained by regression against the true per-frame error. This is the linear-probe pattern imported from self-supervised learning. It is the estimator that should win, since it is the only one that has actually seen what error looks like.

### How do we grade a confidence score?

It is easy to produce a number that correlates with error. It is harder to say whether that number is useful, and this is where the project's evaluation design is more careful than most.

The target is per-frame error, root-aligned so that global translation does not swamp articulation:

$$e(t) = \frac{1}{21}\sum_j\|(\hat p_j - \hat p_0) - (p_j - p_0)\|_2$$

alongside a Procrustes-aligned variant that removes rotation and scale entirely. The gpa between the two is itself informative, as when it is large, the hand's shape is right and its orientation is wrong, which is a different failure than getting the fingers wrong.

Then, Spearman's ρ rather than Pearson's r, because what matters is whether the score ranks frames correctly, not whether it predicts millimetres — a wildly nonlinear but monotone score is perfectly good for filtering. Sparsification curves, which drop the highest-scoring fraction of frames and watch the mean error of the survivors fall, compared against an oracle that drops by true error; the normalized area between the two curves,

$$\mathrm{AUSE} = \frac{1}{E(0)}\int_0^1\big(E(f)-E^*(f)\big)\,df$$

is the single most honest number in the project, because a useless score produces a flat curve and a perfect one hugs the oracle. AUROC for the binary question of whether a frame will exceed a grasp-breaking threshold, computed through the Mann-Whitney rank identity rather than a library call. And finally the experiment that connects everything back to robotics: keep the most confident fraction, and report how much cleaner the surviving demonstration data actually is.

Crucially, all of it is stratified by occlusion severity, frames binned by how much of the object the hand covers. A confidence score that worked only on easy frames would be worthless. The stratification is what turns "we have a score" into "we know where it works."

