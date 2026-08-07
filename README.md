# HaMeR confidence

We want robots to learn manipulation from human video. Let's say, we film a hand picking up a mug; a pose reconstructor (HaMeR) turns each frame into 21 3D joint positions; these trajectories become demonstration data.

However, HaMeR could be confidently wrong exactly when it matters. Hand-object occlusion, in this case, fingers wrapped around the mug, is both the moment that carries the manipulation information and the moment the reconstructor has the least to see. Even worse, nothing in the pipeline knows this. Every frame arrives labeled "here is the hand," with no indication that half of them are fiction.

So the question becomes: can we compute a number, per frame, that perdicts whether the reconstrution is wrong without access to ground truth, while being cheap enough to run over a whole dataset?

If so, actions will change from imitating everything to imitate only what's trustworthy.

**The Vision**
We can't check correctness without ground truth. However, we CAN check consistently, and inconsistency, which is essentially the evidence of error.

- Consistency across perturbations: run the model on slightly different crops. If small input changes swing the output a lot, the model isn't confident.

- Consistency across time: real hands move smoothly. If the reconstruction teleports between adjacent frames, at least one is wrong.

- Consistency across models: if a completely different architecture sees the same hand somewhere else, at least one is wrong. 

None of these need labels, and so the whole trick is to subsitute self-agreement to correctness, which is free!

**Cool Estimators we have**
1. Ensemble disagreement (MC-dropout): run the model $M$ times with jittered crops or dropout active, take per-joint spread

$$u(t) = \frac{1}{21}\sum_j \|\mathrm{std}_m\,\hat p_j^{(m)}(t)\|_2$$

Bayesian story: dropout samples approximate posterior draws, so their variance approximates epistemic uncertainty.

2. Temporal jitter (free!): compare each frame to a median-filtered version of its own neighbors:

$$\tilde p_j(t) = \underset{|t'-t|\le w/2}{\mathrm{median}}\ \hat p_j(t'), \qquad
u(t) = \frac{1}{21}\sum_j \left\|\hat p_j(t) - \tilde p_j(t)\right\|_2$$

Raw frame-to-frame velocity flags fast motion as suspicious (sus sus sus !). Distance to a local median flags only departures from the trajectory, and the median is robust to the outlier frane we're currently scoring, so a bad frame wouldn't corrupt its own reference. 

3. Reporjection residual: project 3D prediction back into the image and compare with an independent 2D detector.

$$\pi(p) = \frac{(Kp)_{xy}}{(Kp)_z}, \qquad
u(t) = \sum_j w_j \left\|\pi(\hat p_j(t)) - d_j(t)\right\|_2$$

The leverage is decorrelated failure modes. Compared to HaMeR, MediaPipe is a different family with different weaknesses. Agreement between two systems that fail differently is much stronger evidence than either alone.

4. Learned head: MLP on frozen ViT features trained with L1 against true MPJPE, Softplus output since error can't be negative. 

**Measuring whether a confidence score is good**

$$e(t) = \frac{1}{21}\sum_j\|(\hat p_j - \hat p_0) - (p_j - p_0)\|_2$$

Root-aligning kills global translation so articulation error dominates. We also compute  PA-MPJPE, which first solves orthogonal Procrustes ($H=\hat P^\top P$, SVD, $R = VDU^\top$ with $D=\mathrm{diag}(1,1,\mathrm{sign}\det)$ to block reflections). If the gap between the two is large, this would mean the hand shape is right but orientation is wrong.

**Does the score track the error?**

- Spearman ρ: we care about ranking frames not predicting error magnitudes. A score that's monotonically related to error but wildly nonlinear is perfectly useful for filtering.

Sparsification / AUSE: the most honest single number. Drop the top $f$ fraction by score, measure mean error of the survivors → $E(f)$. Do the same sorted by true error → the oracle $E^*(f)$. Then

$$\mathrm{AUSE} = \frac{1}{E(0)}\int_0^1\big(E(f)-E^*(f)\big)\,df$$

A useless score gives a flat curve (removing frames doesn't help). A perfect one hugs the oracle. AUSE is the normalized gap between you and the best possible.

AUROC for "will this frame exceed 20 mm," computed via the Mann-Whitney rank identity:

$$\mathrm{AUC} = \frac{\sum_{i\in\text{pos}} r_i - \frac{n_+(n_++1)}{2}}{n_+n_-}$$

That's exactly equal to the ROC area, which is why we get it from rankdata with no sklearn.

Filtering experiment: the one that connects to robotics -- keep the most-confident fraction, report how much cleaner the surviving data is.

And everything stratified by occlusion severity, which is what turns "we have a score" into "we know where it works."

## How this makes sense
The learned head looks at one frame's features and must memorize a mapping to error, from a noisy target, with limited training data. The consistency signals instead exploit structure across observations, smoothness over time, agreement across model families. That information isn't in a single frame's features at all. We can't learn our way to it from the frame alone.