# Course reference for the classifier (MIE1050 Lecture 8)

Source: `RELEVANT COURSE CONTENT.pdf` (Lecture 8, "Data Analysis", plus the assignment slides).
Use **only these methods, in the lecture's form**, so the write-up can cite the lecture for each
step. Anything not in the lecture (the 1D CNN) is kept as small as possible.

## What the assignment asks for (slides 69–72)
- **Part 3:** implement **KNN, naive Bayes, a 1D CNN, and a decision tree or random forest**.
  Compare them with a **simple heuristic baseline**, such as a threshold on peak acceleration.
  Apply **PCA or wrapper feature selection** and report which features survive.
- **Part 4:** run the algorithm on the board.
- **Deliverables:** (1) a classification algorithm in MATLAB or Python; (2) board firmware;
  (3) a 2–4 page summary of the method, why it was chosen, and the problems that remain.
  The slide says "occupant status based on the 4 input parameters". That wording looks copied
  from another assignment; we read it as "which elevator", from the ride features.

## Workflow the lecture prescribes (slide 68 summary)
1. Collect training and verification data.
2. Prepare the data: **scale it**, and remove outliers where possible.
3. Reduce dimensionality with **feature selection or feature extraction (PCA)**.
4. Build a model (KNN, naive Bayes or other).
5. Check its fitness. For classification, report error rates (accuracy, plus a confusion matrix).
6. Update the model as new data comes in.

Data sets (slide 22): **training** (labelled), **verification** (to tune and score), and
**test** (used once at the end). Our versions:
- Training and verification = leave-one-session-out CV.
- Test = the extra session per building that the user is holding back.

Slide 58 lesson: random-forest out-of-bag accuracy was 99.4%, but only 97.0% on new days,
because readings one minute apart are near-copies. **Score on data from different
sessions/days, not a random split.** This justifies leave-one-session-out.

## Heuristics / baseline (slides 13, 58)
- A heuristic is a hand-set rule, e.g. "any value greater than x is classified as R".
- Slide 58 baseline: one threshold (light > 365 lux) scored 99.0% and almost matched the forest.
  Always report a baseline like this.
- Ours: thresholds on **peak (launch) acceleration**, the assignment's own example, set halfway
  between neighbouring class medians in the training folds.

## KNN (slides 26–32)
- Assumption: similar points lie close together in feature space.
- **Euclidean distance**; the class is the **mode of the K nearest** points.
- **Scale every feature to a similar range (e.g. 0–1, min–max)** first, or the large-range
  features dominate the distance.
- **K should be odd.** K = 1 is unstable; a K that is too large makes everything one class.
- Weaknesses: slow with many samples or features, and hurt by the curse of dimensionality.

## Naive Bayes (slides 33–48)
- Bayes: P(H|E) = P(E|H)·P(H) / P(E). Multiply the per-feature probabilities, assuming the
  features are independent ("naive"), and pick the class with the highest product.
- The lecture example classifies elevators from acceleration and deceleration. It
  **discretizes** each feature into bins (Super slow … Super fast) and counts probabilities.
  Priors are assumed equal there.
- Continuous version: Gaussian naive Bayes, which uses a per-class normal distribution in
  place of the bin counts. Same idea, no bin-width choice.
- Naive = independent features. Our motion features are strongly correlated (r 0.88–0.98),
  which breaks that assumption. Feature selection fixes it.
- The probabilities don't sum to 1 (slide 48) because P(E) is left out. The highest one is
  still the answer, and normalizing gives a confidence.

## Decision tree / random forest (slides 49–58)
- Tree: each internal node is a feature threshold, each leaf a class. It is trained
  **greedily to maximize information gain**, where entropy = Σ −pᵢ log₂ pᵢ (max = log₂ n classes).
  → `DecisionTreeClassifier(criterion="entropy")`.
- Greedy training is not guaranteed optimal, and one tree is unstable (it changes with the seed).
- Random forest: **bootstrap** samples (drawn with replacement), plus a **random subset of
  features** at each split, then a **majority vote** across trees. The lecture example uses 100
  trees, each split limited to a few features (sqrt of 5 ≈ 2).
- Report **feature importance (%)** and the vote share as the confidence (e.g. "65 of 100 trees").

## Dimensionality reduction (slides 59–67)
- **Curse of dimensionality**: the training data needed grows exponentially with the number of
  features. That matters for us: 15 features and only about 25–50 rides per class.
- **Feature selection** picks a subset. In the **wrapper** method you train on the training
  set, score each subset's error on the verification set, and keep the best subset. It is the
  most accurate method but slow.
- **Feature extraction** combines features. **PCA** takes the covariance matrix, then its
  eigenvectors (new axes) and eigenvalues (the variance along each); drop the axes with small
  eigenvalues.
- The assignment needs only one ("PCA **or** wrapper"). We use **wrapper (forward selection)**,
  because it reports by name which features survive.

## 1D CNN (not in the lecture)
- Lecture 8 only mentions neural networks and deep learning historically (slides 3–5: deep
  learning "learns higher-level features from the raw input").
- So the CNN gets the **raw waveform**, not our hand-made features. Inputs: the vertical
  acceleration around the launch and around the brake, fixed length, two channels.
- Keep it tiny (two conv layers, then pooling and a softmax), because there are only about
  140 rides.

## Rules for our code
- Every learned step (scaling, feature selection, model) is fitted **inside each CV fold**,
  using a scikit-learn `Pipeline` + `LeaveOneGroupOut(groups=session)`.
- The held-out sessions are scored once, after the model is frozen.
- Plain, commented code; one notebook; no extensions beyond what the assignment asks for.
