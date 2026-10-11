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
