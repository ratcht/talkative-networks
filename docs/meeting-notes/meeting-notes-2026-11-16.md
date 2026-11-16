# Communicative Ensembles: Meeting Notes

**Presenters:** Samyar Goodarzi & Ali Al Hamadani
**Date:** 2026-11-16

## Updates Since Last Meeting

### Finer-Grained Agreement Metrics

- Last meeting it was pointed out that our agreement rate may be too coarse.
  It compares only the top answer of each pair of experts:

$$
\text{agreement} = \operatorname*{mean}_n \; \operatorname*{mean}_{i < j} \, \big[\, \hat y_{n,i} = \hat y_{n,j} \,\big],
$$

  where $\hat y_{n,i}$ is expert $i$'s most likely class on image $n$.

- We have generalised it to the **top-$k$ overlap**. Write
  $\operatorname{top}_k(n, i)$ for the set of expert $i$'s $k$ most likely
  classes on image $n$:

$$
\text{top-}k\text{ overlap} = \operatorname*{mean}_n \; \operatorname*{mean}_{i < j} \, \frac{\lvert \operatorname{top}_k(n, i) \cap \operatorname{top}_k(n, j) \rvert}{k} .
$$

  In words:
    - For every image and every pair of experts, take each expert's shortlist
      of its $k$ most likely classes.
    - Count how many classes the two shortlists share, and divide by $k$.
    - Average over pairs of experts, then over images.

  At $k = 1$ it is the agreement rate. It is logged for $k = 2$ and $3$, before
  and after communication.
- How to read it:
    - **Order within a shortlist doesn't count.** An expert answering shirt
      with coat second and one answering coat with shirt second share their
      whole top 2. Read with the agreement rate, it separates experts that
      disagree on everything from experts that consider the same few classes
      and pick differently.
    - **It isn't a looser agreement.** Experts can agree on the answer more
      often than on the runner-up, so the top-2 overlap can sit below the
      agreement rate.
    - **Chance is $k / C$**, for $C$ classes: 0.2 and 0.3 with 10 classes.
    - **It uses only the ranking, not the probabilities.** It still sees
      runners-up that an expert gives almost no probability. On Fashion-MNIST
      the second class gets a median probability of 0.006, yet two experts
      trained from different seeds share 0.84 of their top 2 before
      communication.

### Negative Correlation at $\lambda = 0$: No Error in the Code

- Last meeting a discrepancy was pointed out. At $\lambda = 0$ (no penalty),
  the negative-correlation experts should be ordinary independent experts, yet
  they looked unlike the baseline pool:
    - individual accuracy 89.7% against 84.3%;
    - agreement 94.1% against 81.8%.

  We were asked to check the code for an error.
- There is no error. The two pools are generated differently:

| | Baseline pool | Negative Corr., $\lambda = 0$ |
|---|---|---|
| Class mix | each expert its own, from Dirichlet($\alpha = 1$) | every class equally (no skew) |
| Sampling | with replacement, by the expert's class mix | every image once per epoch |
| Batch order | each expert its own | shared: all experts train on the same batches |
| Initialisation | each expert its own | each expert its own |

- So at $\lambda = 0$, initialisation is the only thing that differs between
  the experts:
    - Each sees every class in equal proportion, so each is more accurate on
      its own.
    - All see the same batches in the same order, so they agree more.
- Baseline experts differ in class mix, batch order and initialisation.

### Directional Metrics for Queries and Keys

- New metric, the **Q/K cosine gap**: how far apart each receiver's query puts
  the senders, by direction alone. With $q_{n,r}$ receiver $r$'s query and
  $k_{n,s}$ sender $s$'s key on image $n$ ($s$ includes $r$ itself):

$$
\text{Q/K cosine gap} = \operatorname*{mean}_{n,r} \Big( \max_s \cos(q_{n,r}, k_{n,s}) - \min_s \cos(q_{n,r}, k_{n,s}) \Big) .
$$

  It lies between 0 and 2:
    - near 0, a receiver's query sits at the same angle to every sender's key,
      so direction can't tell the senders apart;
    - near 2, the query points along one sender's key and away from another's.
- It is logged every epoch, so it shows how the directions change over
  training:
    - a growing gap means the queries and keys are turning to separate the
      senders;
    - a flat gap means only their lengths can sharpen the attention.
- Read with the metrics already logged:
    - **Query and key norms**: sharpening by length. Falling attention entropy
      with a flat gap and growing norms is sharpening by length; a growing gap
      is sharpening by direction.
    - **Route flip rate**: the share of (image, receiver) pairs whose
      most-heard sender changed since the previous epoch. 0 means routing has
      frozen; 2/3 (with 3 experts) is what re-drawing it at random would give.
    - **Routing mode share**: the share of images routed the most common way.
      1 means the same wiring for every image.

### Class Skew, and an Alternative: Disjoint Class Subsets

- Last meeting there was a concern that the Dirichlet class skew doesn't
  sample with replacement. We have checked that it does:
    - Each expert gets its own class mix, $\pi \sim \operatorname{Dirichlet}(\alpha
      \mathbf{1})$ with $\alpha = 1$.
    - An epoch is $N$ independent draws, where $N$ is the size of the expert
      split. Each draw picks a class $c$ with probability $\pi_c$, then an image
      of that class uniformly at random. The image goes back before the next
      draw.
    - So image $n$ is drawn with probability $\pi_{y_n} / N_{y_n}$, where
      $N_{y_n}$ is the number of images of its class.
    - An expert sees its favoured classes many times an epoch, repeats images,
      and rarely sees the rest. Every expert still takes the same number of
      steps.
- Following the professor's recommendation, we have also built an alternative
  oversampling mechanism, **disjoint class subsets**:
    - For each trial seed, the classes are shuffled and split into $M$ disjoint
      subsets, as evenly as they divide: 4, 3 and 3 of 10 classes for
      $M = 3$. Each expert owns one subset.
    - In place of a Dirichlet mix, each expert's class distribution is
      "one-hot" over the subsets: it sees only its own classes.
    - A slack parameter $\varepsilon$ softens this, in the manner of
      $\varepsilon$-greedy. Each draw comes from the expert's own classes with
      probability $1 - \varepsilon$, and from the rest of the data with
      probability $\varepsilon$, uniformly over images in each case. With
      $S_i$ the images of expert $i$'s classes:

$$
P(\text{draw image } n) =
\begin{cases}
(1 - \varepsilon) / \lvert S_i \rvert & \text{if } n \in S_i, \\
\varepsilon / (N - \lvert S_i \rvert) & \text{otherwise.}
\end{cases}
$$

    - At $\varepsilon = 0$ an expert never sees another class. As
      $\varepsilon$ rises its draws approach uniform over the whole split.
    - Sampling is still with replacement, $N$ draws an epoch, as with class
      skew.
- A control, **data shards**, gives each expert its own disjoint share of the
  images (40%, 30% and 30%), with every class in each share. It has the same
  amount of data per expert as the disjoint subsets, but no class
  specialisation.
- The study comparing these mechanisms is still running and isn't reported
  here.

## Summary of Experiments

- Normalising the query and key stops attention from collapsing onto one
  sender. Routing then varies from image to image, but accuracy doesn't move.
- Training each expert on its own loss gives a slightly better ensemble than
  training on the ensemble's loss. Self-Routing gains just as much, so the gain
  doesn't come from communication.
- In both studies the gains come from each expert's own channel, not from
  hearing the others.

## Shared Setup

- Both studies reuse last meeting's baseline:
    - Each trial seed generates a fresh pool of 3 experts with class skew
      (Dirichlet $\alpha = 1$) on Fashion-MNIST. The experts are then frozen,
      and only the channel trains.
    - Learning rate $10^{-3}$, taper $r = 8$, Q and K of size 16, V of size 64.
    - Two architectures from last meeting: **Linear**, and **Non-Linear Encoder
      & Decoder**, each run forwards and backwards.
- Four mixers:
    - **Attention-Routing**
    - **QK-Norm Attention-Routing** (new, below)
    - **Self-Routing**: each expert hears only its own message
    - **Mean-Routing**: each expert hears the plain average of every message
- Both studies are smaller than last meeting's: **5 trial seeds (0–4) and 10
  joint epochs**, against 15 and 30.
    - 5 seeds can't separate differences under about 0.2 points.
    - The accuracies aren't comparable to last meeting's overview table. Arms
      are compared within each study, paired by trial seed.
- Same scoring as before: the median test accuracy over the 5 epochs around
  each run's best `joint_val` epoch.
- Every $\pm$ below is a standard error over the 5 trial seeds.

## Experimentation: QK-Norm Attention

### Setup

- Last meeting, two problems with plain Attention-Routing:
    - It collapsed onto one sender when the sender was linear.
    - It never beat Self-Routing.
- Question: if attention can't collapse, does it start using the other
  experts' messages?
- **QK-Norm Attention-Routing** divides the query and key by their RMS before
  the dot product, with no learned gain. It adds no parameters.
    - It is the same computation as last meeting's Cosine-Attention-Routing.
      That was run forwards on the Linear channel only; it now runs in both
      directions and at both architectures.
    - After the norm, the attention logit depends only on the angle between
      query and key:

$$
w_{ij} = \operatorname{softmax}_j\!\left(\frac{\hat q_i \cdot \hat k_j}{\sqrt{d_k}}\right)
       = \operatorname{softmax}_j\big(4 \cos(q_i, k_j)\big), \qquad d_k = 16,
$$

  so every logit lies in $[-4, 4]$.

- Sharpening is measured both ways: by length with the query and key norms,
  and by direction with the Q/K cosine gap (see Updates above).

### Results

#### QK-Norm Stops the Collapse

| Architecture | Largest Attn. Weight<br>Attn / QK-Norm | Attention Entropy<br>Attn / QK-Norm | Routing Mode Share<br>Attn / QK-Norm |
|---|---|---|---|
| *Forwards* | | | |
| Linear | 1.00 / 0.63 | 0.01 / 0.83 | 82% / 33% |
| Non-Linear Encoder & Decoder | 0.98 / 0.64 | 0.06 / 0.81 | 80% / 59% |
| *Backwards* | | | |
| Linear | 0.99 / 0.65 | 0.02 / 0.80 | 66% / 50% |
| Non-Linear Encoder & Decoder | 0.92 / 0.64 | 0.18 / 0.80 | 91% / 42% |

*Uniform attention over 3 senders has a largest weight of 1/3 and an entropy of
log 3 ≈ 1.10. Routing mode share is the share of images routed the most common
way.*

- Plain attention collapses within the first epoch and stays collapsed. It
  routes most images the same way, and those routes barely change between
  epochs: 5–14% of routes flip per epoch.
- With QK-Norm, the largest weight settles near 0.64 in every setting. Routing
  varies more with the image and keeps changing: 12–27% of routes flip per
  epoch.
- Why it settles at 0.64:
    - The logit is $4\cos$, and the cosine gap stays at about 0.5 (next
      table), so the best sender's logit is about 2 above the worst's.
    - With the third sender halfway between, the best gets
      $e^2 / (e^2 + e + 1) \approx 0.66$ of the attention.

#### Plain Attention Sharpens Mostly Through Length, Partly Through Direction

| Architecture | Q/K Cosine Gap<br>Attn / QK-Norm | Query Norm<br>Attn / QK-Norm |
|---|---|---|
| *Forwards* | | |
| Linear | 0.87 / 0.45 | 26.7 / 19.6 |
| Non-Linear Encoder & Decoder | 0.76 / 0.50 | 15.7 / 11.3 |
| *Backwards* | | |
| Linear | 0.82 / 0.50 | 20.1 / 18.4 |
| Non-Linear Encoder & Decoder | 0.73 / 0.48 | 12.2 / 10.2 |

*Before training: cosine gap 0.39–0.42, query norm 1.0–1.7. Norms are measured
on the encoder's output. Under QK-Norm they still grow, but the norm removes
them before the attention.*

- Under plain attention:
    - query norms grow 10–20×;
    - the cosine gap roughly doubles, from about 0.4 to 0.73–0.87.
- Both raise the logits, but length is the larger factor. This fits last
  meeting's result that removing length removes the fixed routing.
- Under QK-Norm, the cosine gap rises to 0.45–0.50 in the first epoch and then
  stays flat.

#### Softer Routing Still Doesn't Help Accuracy

| Architecture | QK-Norm $-$ Attn | QK-Norm $-$ Self | QK-Norm $-$ Mean | Attn $-$ Self | Mean $-$ Self |
|---|---|---|---|---|---|
| *Forwards* | | | | | |
| Linear | +0.09 ± 0.07 | +0.10 ± 0.03 | −0.02 ± 0.09 | 0.00 ± 0.06 | +0.11 ± 0.08 |
| Non-Linear Encoder & Decoder | −0.02 ± 0.06 | −0.04 ± 0.03 | −0.06 ± 0.03 | −0.02 ± 0.06 | +0.02 ± 0.03 |
| *Backwards* | | | | | |
| Linear | +0.08 ± 0.05 | −0.01 ± 0.05 | −0.12 ± 0.08 | −0.10 ± 0.05 | +0.10 ± 0.07 |
| Non-Linear Encoder & Decoder | −0.04 ± 0.06 | +0.04 ± 0.03 | −0.09 ± 0.07 | +0.08 ± 0.05 | +0.14 ± 0.05 |

*Difference in test accuracy (percentage points) ± standard error, paired by
trial seed.*

- No difference reaches 0.2 points, and no mixer leads in every setting. This
  repeats last meeting's Cosine-Attention-Routing result, now also backwards
  and on the non-linear channel.
- The two largest differences are 2 of 24 tests, about what chance gives, and
  neither repeats in the other settings:
    - Linear forwards, QK-Norm $-$ Self: +0.10 ± 0.03, ahead on 5 of 5 seeds.
    - Non-linear backwards, Mean $-$ Self: +0.14 ± 0.05.
- Receivers don't pick right senders more often than chance, with either
  mixer:
    - Take the sender each receiver listens to most, itself excluded. That
      sender was right before communication on 82–85% of `joint_val` images.
    - The experts' own accuracy is 83.9%, which is what listening at random
      would score.

## Experimentation: Joint vs Individual Loss

### Setup

- Every run so far trained on the **joint loss**, which scores only the
  ensemble's averaged answer. An expert can stay wrong on an image as long as
  the average is right.
- The **individual loss** scores each expert's own answer, so every expert has
  to be right itself:

$$
L_\text{joint} = -\log\Big(\frac{1}{M}\sum_j p_j[y]\Big),
\qquad
L_\text{individual} = \frac{1}{M}\sum_i -\log p_i[y],
$$

  where $p_i[y]$ is expert $i$'s probability on the true class after
  communication, and $M = 3$.

- The grid is the same as the QK-Norm study, run once with each loss. Two runs
  at the same arm and trial seed differ only in the loss.
- Bagging is the same under both losses for a given trial seed. It moves by 7.5
  points between seeds (82.13–89.66), so the losses are compared paired by
  seed.
- Two questions:
    - Does the individual loss give a better ensemble?
    - Does hearing the others help more under it? This is measured as a
      mixer's lead over Self-Routing trained with the same loss.

### Results

#### The Individual Loss Gives a Slightly Better Ensemble

| Mixer | Joint | Individual | Individual $-$ Joint |
|---|---|---|---|
| Attention-Routing | +2.58 | +2.68 | +0.10 ± 0.03 |
| QK-Norm Attention-Routing | +2.61 | +2.78 | +0.17 ± 0.06 |
| Self-Routing | +2.59 | +2.76 | +0.18 ± 0.03 |
| Mean-Routing | +2.68 | +2.75 | +0.07 ± 0.05 |

*Gain over bagging in test accuracy (percentage points), pooled: each seed is
averaged over the four direction × architecture settings, then averaged over
seeds. A gain over bagging carries bagging's own seed spread (SE ≈ 1.25). The
paired difference in the last column removes it.*

- The individual loss scores higher on 15 of 16 arms on test, and on all 16 on
  `joint_val` (+0.05 to +0.23).
- 13 of the 16 best configurations use the individual loss. The best few are
  within 0.1 points of each other, closer than 5 seeds can separate.

#### …But Not Through Communication

| Mixer | Joint | Individual | Individual $-$ Joint |
|---|---|---|---|
| Attention-Routing | −0.01 ± 0.05 | −0.08 ± 0.05 | −0.07 ± 0.05 |
| QK-Norm Attention-Routing | +0.02 ± 0.01 | +0.02 ± 0.05 | 0.00 ± 0.06 |
| Mean-Routing | +0.09 ± 0.03 | −0.02 ± 0.05 | −0.11 ± 0.06 |

*Lead over Self-Routing in test accuracy (percentage points), pooled as above.*

- Self-Routing gains as much from the individual loss as the mixers that hear
  the others, so their lead over it doesn't grow. No pooled value is positive.
- The same test on the experts' own accuracy is near 0 for every mixer.

#### Experts Get Better and More Alike, and the Two Nearly Cancel

| | Attention | QK-Norm | Self | Mean |
|---|---|---|---|---|
| Avg Individual Expert Accuracy | +1.45 ± 0.44 | +1.41 ± 0.51 | +1.44 ± 0.43 | +1.33 ± 0.49 |
| Agreement Rate | +4.76 ± 1.13 | +4.65 ± 1.29 | +4.74 ± 1.17 | +4.68 ± 1.30 |
| All three right | +5.27 ± 1.45 | +5.16 ± 1.65 | +5.28 ± 1.48 | +5.08 ± 1.65 |
| Some right, average right | −5.15 ± 1.47 | −4.98 ± 1.64 | −5.09 ± 1.46 | −4.96 ± 1.67 |
| Some right, average wrong | −1.45 ± 0.15 | −1.49 ± 0.21 | −1.47 ± 0.23 | −1.48 ± 0.19 |
| None right | +1.32 ± 0.19 | +1.31 ± 0.20 | +1.30 ± 0.21 | +1.41 ± 0.22 |

*Individual $-$ joint on test (percentage points), pooled. The last four rows
sort test images by how many experts answer correctly and whether the averaged
answer is right. Under either loss these four shares sum to 100%, apart from
a ≈0.01% cell where no expert is right but the average is.*

- The individual loss raises each expert's own accuracy by about 1.4 points,
  and the experts agree about 4.7 points more, in every mixer.
- About 5 points move from "some right, average right" to "all three right".
  That changes none of the ensemble's answers.
- 1.5 points leave "some right, average wrong", images an expert knew but the
  vote lost. But 1.3 points join "none right", images every expert now gets
  wrong together. The two nearly cancel, leaving +0.07 to +0.18 for the
  ensemble.
- The weakest trial seed (bagging 82.1) gains 3.2–3.4 points of expert
  accuracy, against about 1 for the others.
- This is the same pattern as last meeting's Mean-Routing and degenerate
  cases: making experts more alike raises their own accuracy far more than the
  ensemble's.

#### Why: Under the Joint Loss, the Average Covers for a Wrong Expert (Hypothesis)

Each loss sends this gradient to expert $i$'s logits $z_i$:

$$
\text{joint:}\quad \frac{\partial L}{\partial z_i} = w_i\,\big(p_i - \text{onehot}(y)\big),
\quad w_i = \frac{p_i[y]}{\sum_j p_j[y]};
\qquad
\text{individual:}\quad \frac{\partial L}{\partial z_i} = \frac{1}{M}\big(p_i - \text{onehot}(y)\big).
$$

- Under the joint loss, an expert's error is scaled by $w_i$, its share of the
  experts' total probability on the true class.
    - An expert that is wrong while the others are right gets almost no
      gradient. For example, with $p_i[y] = 0.05$ and the other two at 0.9,
      $w_i \approx 0.03$.
- Under the individual loss, every expert's error is scaled by $1/M$, so the
  wrong expert is pushed hardest.
- The data fits this:
    - Each expert's own accuracy rises.
    - The weakest experts gain most.
    - The decoders write 30–55% more into each expert.
    - Self-Routing gains as much as the mixers that communicate, because the
      fix happens in each expert's own channel.
- Not measured directly yet: logging $w_i$ under the joint loss should show it
  near 0 on the images expert $i$ gets wrong.
