# HW1 — PAC vs Agnostic PAC

Statistical and Machine Learning — Université de Haute-Alsace (M2 IM / IMDS / MIAGE)

## Goal

Show with experiments the difference between **PAC** learning (clean labels) and **Agnostic PAC** learning (noisy labels), using threshold classifiers on `[0, 1]`:

```
h_t(x) = 1 if x > t, else 0
```

## Files

- `HW1_PAC_vs_Agnostic_PAC.ipynb` — the notebook with the code, plots and comments
- `main (1).pdf` — the homework instructions

## How to run

You need Python with `numpy` and `matplotlib`:

```
pip install numpy matplotlib jupyter
jupyter notebook HW1_PAC_vs_Agnostic_PAC.ipynb
```

Then click **Run All**. No machine learning library is used.

## Parameters

| Parameter | Value |
|---|---|
| True threshold θ | 0.3 |
| Accuracy ε | 0.05 |
| Noise rate η | 0.2 |
| Training sizes m | 5, 10, 20, 50, 100, 200 |
| Repetitions | 500 |
| Test set size | 10 000 |

## Notebook sections

1. **Data generation**: uniform points in `[0, 1]`, labelled by θ, with each label flipped with probability η in the noisy case.
2. **ERM**: try `t = 0` and every training point as the threshold, and keep the one with the smallest training error.
3. **Performance evaluation**: training error vs true error, which is estimated on a large test set.
4. **PAC learning**: estimate `P(L(h) > ε)` for each m.
5. **Agnostic PAC**: the same experiment with noise, then a comparison of the two settings.

## Main results

- **PAC (no noise):** the training error is always 0 and the true error goes to 0 as m grows.
- **Agnostic PAC (noise):** the training error stays around η and the true error goes down to η = 0.2, never to 0. The probability of a bad hypothesis gets smaller more slowly than without noise.

---

# HW2 — Uniform Convergence and the Limits of Learning

Statistical and Machine Learning — Université de Haute-Alsace (M2 IM / IMDS / MIAGE)

## Goal

Study, by Monte-Carlo simulation, three ideas about the limits of learning:

1. **Uniform convergence** — how the empirical risk `L_S(h)` fluctuates around the true risk `L_D(h)`, and how this is controlled by the sample size `m` and the hypothesis-class size `|H|`.
2. **No-Free-Lunch** — with completely random targets, a learner cannot predict the labels of inputs it has never seen.
3. **Inductive bias** — restricting the target and the hypothesis class to a structured family (thresholds) makes generalization possible.

## Files

- `hm2/HW2.ipynb` — the notebook with the code, plots and comments

## How to run

You need Python with `numpy` and `matplotlib`:

```
pip install numpy matplotlib jupyter
jupyter notebook hm2/HW2.ipynb
```

Then click **Run All**. No machine learning library is used.

## Parameters

| Parameter | Value |
|---|---|
| True risks `p_j` | 0.3 (single), or uniform in [0, 1] (class) |
| Accuracy ε | 0.1 |
| Confidence δ | 0.1 |
| Class sizes `|H|` | 5, 10, 50, 100, 500 |
| Domain size N | 100 |
| Sample sizes m | swept from 1 up to ~800 |
| Monte-Carlo repetitions | 800 – 20 000 (depending on cost) |

## Notebook sections

**Part I — Uniform convergence (tasks 1–5).** Empirical risk as a Binomial random variable; the deviation probability vs. the Hoeffding bound; the maximum deviation `Δ_S` over the whole class vs. the union bound `2|H|e^(-2mε²)`; the effect of `|H|`; experimental vs. theoretical sample complexity.

**Part II — No-Free-Lunch (tasks 6–8).** Fraction of unseen inputs vs. `(1-1/N)^m`; true risk of the memorization learner vs. `½(1-1/N)^m`; comparison of the predict-0 / predict-1 / random rules on unseen inputs.

**Part III — Inductive bias (tasks 9–11).** ERM over threshold hypotheses against a threshold target; random targets vs. structured targets; final analysis.

## Main results

- **Concentration:** `L_S(h)` is unbiased and its spread shrinks like `√(p(1-p)/m)`, so it concentrates around `L_D(h)` as m grows.
- **Conservative bounds:** the experimental deviation probabilities stay well below the Hoeffding and union bounds, and the experimental sufficient sample size (m ≈ 200) is smaller than the theoretical one (m ≈ 345).
- **No-Free-Lunch:** with random targets all prediction rules give the same expected risk `½(1-1/N)^m` — unseen labels cannot be inferred from the sample.
- **Inductive bias:** with a threshold target learned by threshold ERM the true risk collapses to ≈ 0 after only a few samples, showing how prior knowledge enables generalization.
