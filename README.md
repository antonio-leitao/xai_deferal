# Explaining Learning to Defer

Code for the paper _Explaining Learning to Defer_.

- `experiments/`: `core.py` (all shared code), `openl2d.py` (the synthetic expert of OpenL2D,
  Alves et al., 2025) and one notebook per experiment.
- `defer/`: the L2D losses and network of Mozannar et al. (2023).
- `data/`: Adult (Becker and Kohavi, 1996), Plant (Rubekin, 2024), Wine (Cortez et al., 2009) and
  COMPAS (`broward/`, Dressel and Farid, 2018).

## Running

```bash
uv sync
uv run jupyter execute --inplace experiments/compas.ipynb
uv run jupyter execute --inplace experiments/validation.ipynb
uv run jupyter execute --inplace experiments/detection.ipynb
uv run jupyter execute --inplace experiments/robustness.ipynb
uv run jupyter execute --inplace experiments/plausibility.ipynb
```

Run them in this order: `detection` and `plausibility` read the runs of `compas` and `validation`.
Runs are cached in `experiments/cache/` and figures and tables written to `experiments/figures/`.
The first execution of `validation` trains 630 models and takes about 2 hours on 8 CPU cores; the
others take minutes, and every later execution reads the cache.

| Notebook       | Paper                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| `compas`       | COMPAS explanations and fitted-expert AUC                                                                     |
| `validation`   | Deferral rate and deferral types vs. expert accuracy; appendix accuracy, coverage and known-expert comparison |
| `detection`    | Uncertainty and expected expert performance at deferred points and counterfactuals; quadrant plots            |
| `robustness`   | Stability of the deferral types under perturbation                                                            |
| `plausibility` | Distance of the counterfactuals to the training data                                                          |

The `EXPERT` cell at the top of `validation`, `detection` and `robustness` selects the expert model
behind expected expert performance: `"fitted"` (default, as in the paper) or `"true"` (the known
synthetic expert, for the appendix comparison). Switching it reads the same cache.
