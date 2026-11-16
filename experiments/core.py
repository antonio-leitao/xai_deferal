"""Shared code for the experiments in the paper.

A run caches only quantities that do not depend on the expert model: the
deferred points, their counterfactuals, the classifier probabilities and the
defer scores at both. Expected expert performance and the deferral types are
computed when a notebook reads the cache (`assign_types`), from either the
known synthetic expert ("true") or the expert fitted by MLE on the training
split ("fitted"). Changing the expert therefore never requires a rerun.

COMPAS has real human predictions and no known expert, so it always uses the
fitted expert.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT):  # openl2d.py, and the defer package of Mozannar et al.
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import openl2d  # noqa: E402

from defer.baselines.lce_surrogate import LceSurrogate  # noqa: E402
from defer.baselines.one_v_all import OVASurrogate  # noqa: E402
from defer.methods.realizable_surrogate import RealizableSurrogate  # noqa: E402
from defer.networks.non_linear_net import NonLinearNet  # noqa: E402

CACHE = HERE / "cache"
FIGURES = HERE / "figures"

DATASETS = ("adult", "plant", "wine")
METHODS = ("realizable_surrogate", "crossentropy_surrogate", "one_vs_all")
ACCURACIES = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
SEEDS = tuple(range(10))
DATASET_LABELS = {"adult": "Adult", "plant": "Plant", "wine": "Wine", "compas": "COMPAS"}
METHOD_LABELS = {
    "realizable_surrogate": "Realizable Surrogate",
    "crossentropy_surrogate": "Cross-Entropy Surrogate",
    "one_vs_all": "One-vs-All",
}
TYPES = ("type_i", "type_ii", "mixed", "neither")

# Training (Appendix, Training parameters).
EPOCHS, TEST_INTERVAL, LEARNING_RATE, BATCH_SIZE = 100, 5, 1e-2, 1000
ALPHA_GRID = [0.0, 0.1, 0.3, 0.5, 0.9, 1.0]


@dataclass(frozen=True)
class Stage:
    distance_fraction: float
    learning_rate: float
    steps: int
    init_scale: float


# Counterfactual search (Appendix, Counterfactual search).
STAGES = (
    Stage(0.5, 0.1, 500, 0.0),
    Stage(0.2, 0.05, 800, 0.0),
    Stage(0.05, 0.03, 1200, 0.02),
    Stage(0.0, 0.01, 1600, 0.10),
)
MARGIN, SEARCH_BATCH, BOUNDARY_STEPS, PATIENCE, GRADIENT_CLIP = 1e-4, 256, 28, 80, 10.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- data


ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                 "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
                 "income"]


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and binary labels of Adult, Plant or Wine, every feature standardized."""
    folder = ROOT / "data"
    if name == "adult":
        # adult.data and adult.test from https://archive.ics.uci.edu/dataset/2/adult; rows with
        # missing values are dropped, "race" is binarized and "education" (coded by
        # "education_num") is dropped; categorical columns are ordinal-encoded.
        parts = [pd.read_csv(folder / "adult/adult.data", header=None),
                 pd.read_csv(folder / "adult/adult.test", header=None, skiprows=1)]
        for part in parts:
            part.columns = ADULT_COLUMNS
            part.replace(" ?", np.nan, inplace=True)
            part.dropna(inplace=True)
        frame = pd.concat(parts, ignore_index=True)
        frame["income"] = frame["income"].apply(lambda v: 1 if (v == " >50K") or (v == " >50K.") else 0)
        frame["binarized_race"] = frame["race"].apply(lambda v: "White" if v == " White" else "Non-White")
        frame = frame[ADULT_COLUMNS[:9] + ["binarized_race"] + ADULT_COLUMNS[9:]].drop(columns=["education", "race"])
        frame["relationship"] = frame["relationship"].apply(lambda v: " Married" if (v == " Wife") or (v == " Husband") else v)
        categorical = ["workclass", "marital_status", "occupation", "relationship", "binarized_race", "sex", "native_country"]
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        frame[categorical] = encoder.fit_transform(frame[categorical])
        label = "income"
    elif name == "plant":
        frame, label = pd.read_csv(folder / "plant_disease.csv"), "disease_present"
    elif name == "wine":  # https://www.kaggle.com/datasets/yasserh/wine-quality-dataset; quality above 5 is positive
        frame = pd.read_csv(folder / "wine.csv")
        frame["quality"] = frame["quality"].apply(lambda v: 1 if (v > 5) else 0)
        frame, label = frame.drop("Id", axis=1), "quality"
    else:
        raise ValueError(name)
    y = frame[label].values
    x = frame.drop([label], axis=1)
    x[x.columns] = StandardScaler().fit_transform(x[x.columns])
    return x, y


class Split:
    """70/10/20 train/validation/test split fixed by the seed, with the shuffled loaders used in training."""

    def __init__(self, x: np.ndarray, y: np.ndarray, human: np.ndarray, seed: int, batch_size: int = BATCH_SIZE):
        x, y, human = torch.from_numpy(x).float(), torch.tensor(y), torch.tensor(human)
        self.d, self.batch_size = x.shape[1], batch_size
        n_train, n_val = int((1 - 0.2 - 0.1) * len(x)), int(0.1 * len(x))
        sizes = [n_train, n_val, len(x) - n_train - n_val]
        self.train_x, self.val_x, self.test_x = torch.utils.data.random_split(
            x, sizes, generator=torch.Generator().manual_seed(seed))
        for name, part in (("train", self.train_x), ("val", self.val_x), ("test", self.test_x)):
            data = torch.utils.data.TensorDataset(x[part.indices], y[part.indices], human[part.indices])
            setattr(self, f"data_{name}", data)
            setattr(self, f"data_{name}_loader", DataLoader(data, batch_size=batch_size, shuffle=True))


def split(x: pd.DataFrame, y: np.ndarray, human: np.ndarray, seed: int) -> Split:
    return Split(x.to_numpy(), y, human, seed)


def load_compas(seed: int):
    """COMPAS with the majority vote of the Mechanical Turk participants as human prediction.

    The scaler is fitted on the training split only. Returns the raw table, the
    scaled features, the split and the scaler.
    """
    folder = ROOT / "data/broward"  # Dressel and Farid (2018)
    raw = pd.read_csv(folder / "BROWARD_CLEAN_SUBSET.csv")
    votes = pd.read_csv(folder / "MTURK_RACE.csv")
    ids = pd.to_numeric(votes.pop("mTurk_code"), errors="coerce")
    votes = votes.loc[ids.notna()].apply(pd.to_numeric, errors="raise")
    votes.index = ids.loc[ids.notna()].astype(int)
    votes = votes.reindex(raw["id"])
    # Entries encode correctness (+1 correct, -1 incorrect).
    correct = votes.apply(lambda row: row.dropna().value_counts().idxmax(), axis=1).to_numpy() == 1
    y = raw["two_year_recid"].to_numpy(dtype=np.int64)
    human = np.where(correct, y, 1 - y).astype(np.int64)
    features = raw.drop(columns=["block_num", "id", "two_year_recid"])
    order = torch.randperm(len(raw), generator=torch.Generator().manual_seed(seed)).numpy()
    n_train = int((1 - 0.2 - 0.1) * len(raw))
    scaler = StandardScaler().fit(features.iloc[order[:n_train]])
    x = pd.DataFrame(scaler.transform(features), columns=features.columns)
    data = split(x, y, human, seed)
    assert np.array_equal(data.train_x.indices, order[:n_train])
    return raw, x, data, scaler


def feature_range(x: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature bounds of the counterfactual search: the observed range."""
    return tuple(torch.as_tensor(v.to_numpy(), dtype=torch.float32).unsqueeze(0) for v in (x.min(), x.max()))


# ------------------------------------------------------------------------ experts


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = 1.0 / (1.0 + np.exp(-np.clip(values, -15, 15)))
    result[values > 15] = 1.0
    result[values < -15] = 0.0
    return result


@dataclass(frozen=True)
class Expert:
    """OpenL2D expert (Eq. openl2d): P(FP | x) and P(FN | x) are logistic in w.x/||w||."""

    fpr_beta: float
    fnr_beta: float
    alpha: float
    w: tuple[float, ...]

    def error_probabilities(self, x: np.ndarray) -> np.ndarray:
        """Columns: P(h(x)=1 | y=0, x) and P(h(x)=0 | y=1, x)."""
        weights = np.asarray(self.w, dtype=float)
        projection = float(self.alpha) * (x @ (weights / np.linalg.norm(weights)))
        return np.column_stack((_sigmoid(self.fpr_beta + projection), _sigmoid(self.fnr_beta - projection)))


def simulate_expert(x: pd.DataFrame, y: np.ndarray, accuracy: float, seed: int):
    """Synthetic OpenL2D expert (alpha = 30) with a random binary weight vector and the requested accuracy.

    Returns the expert and its decisions on (x, y).
    """
    weights = np.random.default_rng(seed).integers(0, 2, size=x.shape[1])
    if not weights.any():
        weights[seed % len(weights)] = 1
    w = weights.astype(float)
    fpr_beta, fnr_beta, human = openl2d.simulate(x, y, w, fpr=1.0 - accuracy, fnr=1.0 - accuracy, alpha=30, seed=seed)
    return Expert(float(fpr_beta), float(fnr_beta), 30.0, tuple(w.tolist())), human


def fit_expert(x: torch.Tensor, y: torch.Tensor, human: torch.Tensor, seed: int, epochs: int = 2000) -> Expert:
    """Regularized MLE of the OpenL2D parameters (Eq. mle, Appendix Expert Simulation)."""
    x = x.double()
    y, human = y.long(), human.long()
    error = (human != y).double()
    rates = [float(error[y == label].mean().clamp(0.01, 0.99)) for label in [0, 1]]
    beta = torch.tensor(np.log(np.array(rates) / (1 - np.array(rates))), requires_grad=True)
    alpha = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    w = torch.randn(x.shape[1], dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    w = (w / w.norm()).requires_grad_()
    parameters = [beta, alpha, w]
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    best_loss, best = float("inf"), None
    for step in range(epochs + 1):
        optimizer.zero_grad()
        projection = alpha * (x @ (w / w.norm().clamp_min(1e-6)))
        logits = torch.where(y == 0, beta[0] + projection, beta[1] - projection)
        loss = F.binary_cross_entropy_with_logits(logits, error, reduction="sum")
        loss = loss + 0.01 * (beta.square().sum() + alpha.square() + w.square().sum())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best = Expert(beta[0].item(), beta[1].item(), alpha.item(), tuple(w.detach().numpy().tolist()))
        if step == epochs:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(min=0.1)
    return best


def synthetic_task(dataset: str, accuracy: float, seed: int):
    """Features, split with the synthetic expert's decisions, and the synthetic expert."""
    x, y = load_dataset(dataset)
    expert, human = simulate_expert(x, y, accuracy, seed)
    return x, split(x, y, human, seed), expert


def expert_path(dataset: str, accuracy: float | None, seed: int) -> Path:
    name = f"compas__seed_{seed:03d}" if dataset == "compas" else f"{dataset}__acc_{accuracy:.3f}__seed_{seed:03d}"
    return CACHE / "experts" / f"{name}.json"


def experts(dataset: str, accuracy: float | None, seed: int) -> dict[str, Expert | None]:
    """The known expert ("true", None for COMPAS) and the MLE fit on the training split ("fitted")."""
    path = expert_path(dataset, accuracy, seed)
    if not path.exists():
        if dataset == "compas":
            true, data = None, load_compas(seed)[2]
        else:
            _, data, true = synthetic_task(dataset, accuracy, seed)
        fitted = fit_expert(*data.data_train.tensors, seed=seed)
        _write_json(path, {"true": None if true is None else asdict(true), "fitted": asdict(fitted)})
    stored = json.loads(path.read_text())
    return {kind: None if params is None else Expert(**params) for kind, params in stored.items()}


def expert_model(dataset: str, accuracy: float | None, seed: int, kind: str) -> Expert:
    """The expert used for the diagnostics: `kind` is "true" or "fitted"."""
    assert kind in ("true", "fitted"), kind
    models = experts(dataset, accuracy, seed)
    return models["fitted"] if models["true"] is None else models[kind]


# ----------------------------------------------------------------------- training


class StableRealizableSurrogate(RealizableSurrogate):
    """Realizable Surrogate with its loss computed in log space (same objective, no overflow)."""

    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        human_correct = hum_preds.eq(data_y)
        row = torch.arange(outputs.shape[0], device=outputs.device)
        true_logits = outputs[row, data_y]
        reject_logits = outputs[:, -1]
        joint_numerator = torch.where(human_correct, torch.logaddexp(true_logits, reject_logits), true_logits)
        joint_loss = torch.logsumexp(outputs, dim=1) - joint_numerator
        classification_loss = torch.logsumexp(outputs[:, :-1], dim=1) - true_logits
        loss = self.alpha * joint_loss + (1.0 - self.alpha) * classification_loss
        return loss.mean() / math.log(2.0)


class StableLceSurrogate(LceSurrogate):
    """Cross-Entropy Surrogate evaluated in log space."""

    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        human_correct = hum_preds.eq(data_y).float()
        machine_weight = self.alpha * human_correct + (1.0 - human_correct)
        log_probabilities = F.log_softmax(outputs, dim=1)
        row = torch.arange(outputs.shape[0], device=outputs.device)
        loss = -human_correct * log_probabilities[:, -1] - machine_weight * log_probabilities[row, data_y]
        return loss.mean() / math.log(2.0)


class StableOVASurrogate(OVASurrogate):
    """One-vs-All Surrogate with an overflow-safe logistic loss."""

    def LogisticLossOVA(self, outputs, y):
        return F.softplus((-1 * y) * outputs) / math.log(2.0)


def train(data, method: str, seed: int):
    """Train an L2D system; alpha and the deferral threshold are selected on the validation split."""
    set_seed(seed)
    model = NonLinearNet(data.d, 3)
    cpu = torch.device("cpu")
    l2d = {
        "realizable_surrogate": lambda: StableRealizableSurrogate(1, 300, model, cpu, True),
        "crossentropy_surrogate": lambda: StableLceSurrogate(1, 300, model, cpu, False),
        "one_vs_all": lambda: StableOVASurrogate(1, 300, model, cpu, False),
    }[method]()
    arguments = dict(
        dataloader_train=data.data_train_loader, dataloader_val=data.data_val_loader,
        dataloader_test=data.data_test_loader, epochs=EPOCHS, optimizer=torch.optim.Adam,
        scheduler=None, lr=LEARNING_RATE, verbose=False, test_interval=TEST_INTERVAL,
    )
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if method == "realizable_surrogate":
            l2d.fit_hyperparam(**arguments, alpha_grid=ALPHA_GRID)
        elif method == "crossentropy_surrogate":
            l2d.fit_hyperparam(**arguments)
        else:
            l2d.fit(**arguments)
    assert all(bool(torch.isfinite(p).all()) for p in l2d.model.parameters()) and np.isfinite(l2d.threshold_rej)
    return l2d, time.perf_counter() - started


def defer_scores(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """p(defer) - p(most likely class); the system defers when this reaches the threshold."""
    probabilities = F.softmax(model(x), dim=1)
    predicted = torch.argmax(probabilities[:, :-1], dim=1)
    return probabilities[:, -1] - probabilities.gather(1, predicted.unsqueeze(1)).squeeze(1)


def class_probabilities(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Classifier posterior p_M(y | x), the defer output excluded."""
    return F.softmax(model(x)[:, :-1], dim=1)


def deferred_test_points(l2d, data, max_points: int | None = None, seed: int = 0):
    """Deferred test points (a seeded subsample of `max_points` if given), their test indices,
    the number deferred, and the system and classifier accuracy on the test split."""
    output = l2d.test(DataLoader(data.data_test, batch_size=data.batch_size, shuffle=False))
    defer = output["defers"].astype(bool)
    system = np.where(defer, output["hum_preds"], output["preds"])
    metrics = {"system_acc": float(np.mean(system == output["labels"])),
               "classifier_all_acc": float(np.mean(output["preds"] == output["labels"]))}
    test_x = data.data_test.tensors[0]
    with torch.no_grad():
        scores = defer_scores(l2d.model, test_x).numpy()
    index = np.flatnonzero(scores >= float(l2d.threshold_rej))
    n_deferred = len(index)
    if max_points is not None and n_deferred > max_points:
        index = np.sort(np.random.default_rng(seed + 13_337).choice(index, size=max_points, replace=False))
    return test_x[index], index, n_deferred, metrics


# ------------------------------------------------------------------------- search


def _gradient_scale(model, x):
    probe = x.detach().clone().requires_grad_(True)
    gradient = torch.autograd.grad(defer_scores(model, probe).sum(), probe)[0]
    return torch.nan_to_num(gradient.abs().amax(dim=1), nan=0.0, posinf=0.0)


def _project(delta, x, lower, upper):
    with torch.no_grad():
        delta.copy_(torch.maximum(torch.minimum(x + delta, upper), lower) - x)


def _optimize_stage(model, x, threshold, scale, lower, upper, stage, generator):
    """Minimize relu(s(x+d) - threshold + margin) + lambda ||d||_1 for one continuation stage."""
    if stage.init_scale > 0:
        delta = torch.randn(x.shape, generator=generator, dtype=x.dtype) * stage.init_scale
    else:
        delta = torch.zeros_like(x)
    delta = delta.detach().clone().requires_grad_(True)
    _project(delta, x, lower, upper)
    weight = scale * stage.distance_fraction
    optimizer = torch.optim.Adam([delta], lr=stage.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(stage.steps, 1), eta_min=stage.learning_rate * 0.05)
    best_delta = torch.zeros_like(x)
    best_l1 = torch.full((len(x),), float("inf"))
    found = torch.zeros(len(x), dtype=torch.bool)
    last_improvement = 0
    for step in range(stage.steps):
        optimizer.zero_grad(set_to_none=True)
        violation = F.relu(defer_scores(model, x + delta) - threshold + MARGIN)
        loss = (violation + weight * delta.abs().sum(dim=1)).sum()
        loss.backward()
        delta.grad.clamp_(-GRADIENT_CLIP, GRADIENT_CLIP)
        optimizer.step()
        _project(delta, x, lower, upper)
        scheduler.step()
        with torch.no_grad():
            valid = defer_scores(model, x + delta) <= threshold - MARGIN
            l1 = delta.abs().sum(dim=1)
            improved = valid & (l1 < best_l1)
            if improved.any():
                best_l1[improved] = l1[improved]
                best_delta[improved] = delta.detach()[improved]
                found |= improved
                last_improvement = step + 1
        if found.all() and step + 1 - last_improvement >= PATIENCE:
            break
    return best_delta, found


def _refine_to_boundary(model, x, delta, threshold):
    """Bisect each factual-to-candidate segment for the nearest valid point."""
    low = torch.zeros((len(x), 1), dtype=x.dtype)
    high = torch.ones((len(x), 1), dtype=x.dtype)
    with torch.no_grad():
        for _ in range(BOUNDARY_STEPS):
            middle = (low + high) / 2
            valid = (defer_scores(model, x + middle * delta) <= threshold - MARGIN).unsqueeze(1)
            high = torch.where(valid, middle, high)
            low = torch.where(valid, low, middle)
    return high * delta


def search_counterfactuals(model, factuals, threshold, lower, upper, seed):
    """Counterfactual search (Eq. cfdefer, Appendix Counterfactual search).

    Returns the perturbations (NaN where none was found), the found flags, the
    stage that found each one (-1 if none) and the wall time in seconds.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    n = len(factuals)
    deltas = torch.full_like(factuals, float("nan"))
    found = np.zeros(n, dtype=bool)
    stage_index = np.full(n, -1, dtype=int)
    generator = torch.Generator()
    generator.manual_seed(seed + 104_729)
    started = time.perf_counter()
    for start in range(0, n, SEARCH_BATCH):
        stop = min(start + SEARCH_BATCH, n)
        rows, x_batch = np.arange(start, stop), factuals[start:stop]
        scales = _gradient_scale(model, x_batch).detach()
        unresolved = np.arange(len(x_batch))
        for index, stage in enumerate(STAGES):
            if len(unresolved) == 0:
                break
            delta, stage_found = _optimize_stage(
                model, x_batch[unresolved], threshold, scales[unresolved], lower, upper, stage, generator)
            stage_found = stage_found.numpy()
            if stage_found.any():
                hit = rows[unresolved[stage_found]]
                deltas[hit] = delta[stage_found].detach()
                found[hit] = True
                stage_index[hit] = index
            unresolved = unresolved[~stage_found]
    mask = torch.from_numpy(found)
    if found.any():
        deltas[mask] = _refine_to_boundary(model, factuals[mask], deltas[mask], threshold)
        with torch.no_grad():
            verified = (defer_scores(model, factuals[mask] + deltas[mask]) <= threshold - MARGIN).numpy()
        invalid = np.flatnonzero(found)[~verified]
        found[invalid], stage_index[invalid] = False, -1
        deltas[invalid] = float("nan")
    return deltas, found, stage_index, time.perf_counter() - started


def pairs_frame(model, factuals: torch.Tensor, deltas: torch.Tensor, found: np.ndarray) -> pd.DataFrame:
    """Expert-independent record of each (factual, counterfactual) pair.

    Columns x{j} and cf{j} hold the points, p0_/p1_ the classifier probabilities
    and score_ the defer scores at the factual (_x) and counterfactual (_cf).
    Counterfactual columns are NaN where the search found none.
    """
    n, d = factuals.shape
    mask = torch.from_numpy(found)
    columns = {"found": found}
    for side, rows in (("x", np.ones(n, dtype=bool)), ("cf", found)):
        values = np.full((n, d), np.nan, dtype=np.float32)
        probabilities = np.full((n, 2), np.nan, dtype=np.float32)
        scores = np.full(n, np.nan, dtype=np.float32)
        if rows.any():
            batch = factuals if side == "x" else factuals[mask] + deltas[mask]
            with torch.no_grad():
                probabilities[rows] = class_probabilities(model, batch).numpy()
                scores[rows] = defer_scores(model, batch).numpy()
            values[rows] = batch.numpy()
        columns.update({f"{side}{j}": values[:, j] for j in range(d)})
        columns.update({f"p0_{side}": probabilities[:, 0], f"p1_{side}": probabilities[:, 1], f"score_{side}": scores})
    return pd.DataFrame(columns)


# ---------------------------------------------------------------------- diagnostics


def points(pairs: pd.DataFrame, side: str) -> np.ndarray:
    """The factual (side="x") or counterfactual (side="cf") points as a float32 array."""
    d = sum(1 for column in pairs.columns if column.startswith("x") and column[1:].isdigit())
    return np.ascontiguousarray(pairs[[f"{side}{j}" for j in range(d)]].to_numpy(np.float32))


def assign_types(pairs: pd.DataFrame, expert: Expert) -> pd.DataFrame:
    """Add machine uncertainty U_M, expected expert performance E_h and the deferral type.

    U_M is the entropy of p_M (Eq. entropy) and E_h = sum_y P(h(x)=y | y, x) p_M(y | x)
    (Eq. exp_perf). Type I: U_M decreases at the counterfactual; Type II: E_h
    decreases; Mixed: both; Neither: none. Pairs without a counterfactual get no type.
    """
    found = pairs["found"].to_numpy(bool)
    out = pairs.copy()
    for side, rows in (("x", slice(None)), ("cf", found)):
        probabilities = out[[f"p0_{side}", f"p1_{side}"]].to_numpy(np.float32)
        out[f"uncertainty_{side}"] = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        error = np.full(len(out), np.nan)
        error[rows] = np.sum(expert.error_probabilities(points(out, side)[rows]) * probabilities[rows], axis=1)
        out[f"expert_performance_{side}"] = 1.0 - error
        out[f"expert_error_{side}"] = error
    type_i = found & (out["uncertainty_x"] > out["uncertainty_cf"]).to_numpy()
    type_ii = found & (out["expert_error_cf"] > out["expert_error_x"]).to_numpy()
    category = np.select([type_i & type_ii, type_i, type_ii], ["mixed", "type_i", "type_ii"], "neither")
    out["category"] = np.where(found, category, None)
    return out


# -------------------------------------------------------------------------- cache


def _file(path: Path, extension: str) -> Path:
    return path.with_name(f"{path.name}.{extension}")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, engine="fastparquet", index=False)
    os.replace(temporary, path)


def save_run(path: Path, pairs: pd.DataFrame, info: dict) -> None:
    """Write `<path>.parquet` and `<path>.json`; the JSON is written last and marks the run complete."""
    _write_parquet(_file(path, "parquet"), pairs)
    _write_json(_file(path, "json"), info)


def load_run(path: Path) -> tuple[pd.DataFrame, dict]:
    return pd.read_parquet(_file(path, "parquet"), engine="fastparquet"), json.loads(_file(path, "json").read_text())


def is_cached(path: Path) -> bool:
    return _file(path, "json").exists()


# ---------------------------------------------------------------------- experiments


def validation_path(dataset: str, method: str, accuracy: float, seed: int) -> Path:
    return CACHE / "validation" / f"{dataset}__{method}__acc_{accuracy:.3f}__seed_{seed:03d}"


def run_validation(dataset: str, method: str, accuracy: float, seed: int) -> Path:
    """One synthetic-expert configuration: train, then search every deferred test point."""
    path = validation_path(dataset, method, accuracy, seed)
    if is_cached(path):
        return path
    x, data, _ = synthetic_task(dataset, accuracy, seed)
    l2d, training_seconds = train(data, method, seed)
    factuals, test_index, n_deferred, metrics = deferred_test_points(l2d, data)
    lower, upper = feature_range(x)
    deltas, found, stage, search_seconds = search_counterfactuals(
        l2d.model, factuals, float(l2d.threshold_rej), lower, upper, seed)
    pairs = pairs_frame(l2d.model, factuals, deltas, found).assign(test_index=test_index, stage=stage)
    info = dict(dataset=dataset, method=method, accuracy=accuracy, seed=seed, n_test=len(data.data_test),
                n_deferred=n_deferred, threshold=float(l2d.threshold_rej), alpha=float(l2d.alpha),
                training_seconds=training_seconds, search_seconds=search_seconds, **metrics)
    save_run(path, pairs, info)
    return path


def compas_path(seed: int) -> Path:
    return CACHE / "compas" / f"seed_{seed:03d}"


def run_compas(seed: int = 0) -> Path:
    """Realizable Surrogate on COMPAS: train, then search every deferred test point."""
    path = compas_path(seed)
    if is_cached(path):
        return path
    set_seed(seed)
    _, x, data, _ = load_compas(seed)
    l2d, training_seconds = train(data, "realizable_surrogate", seed)
    factuals, test_index, n_deferred, metrics = deferred_test_points(l2d, data)
    lower, upper = feature_range(x)
    deltas, found, stage, search_seconds = search_counterfactuals(
        l2d.model, factuals, float(l2d.threshold_rej), lower, upper, seed)
    pairs = pairs_frame(l2d.model, factuals, deltas, found).assign(test_index=test_index, stage=stage)
    info = dict(dataset="compas", method="realizable_surrogate", seed=seed, n_test=len(data.data_test),
                n_deferred=n_deferred, threshold=float(l2d.threshold_rej), alpha=float(l2d.alpha),
                training_seconds=training_seconds, search_seconds=search_seconds, **metrics)
    save_run(path, pairs, info)
    return path


ROBUSTNESS_ACCURACY, ROBUSTNESS_INSTANCES, ROBUSTNESS_DIRECTIONS = 0.75, 110, 8
ROBUSTNESS_RADII = np.linspace(0.1, 0.5, 10)


def robustness_path(dataset: str, method: str, seed: int) -> Path:
    return CACHE / "robustness" / f"{dataset}__{method}__seed_{seed:03d}"


def run_robustness(dataset: str, method: str, seed: int = 0) -> Path:
    """Perturb deferred points around their counterfactual distance and search again.

    For each deferred point x with a counterfactual x + delta, draw
    ROBUSTNESS_DIRECTIONS random directions at each radius r * ||delta||_2, keep
    the perturbed points that still defer and search their counterfactuals.
    Writes `<path>.parquet` (base pairs), `<path>__perturbed.parquet` and `<path>.json`.
    """
    path = robustness_path(dataset, method, seed)
    if is_cached(path):
        return path
    if dataset == "compas":
        set_seed(seed)
        _, x, data, _ = load_compas(seed)
    else:
        x, data, _ = synthetic_task(dataset, ROBUSTNESS_ACCURACY, seed)
    l2d, _ = train(data, method, seed)
    threshold = float(l2d.threshold_rej)
    lower, upper = feature_range(x)
    factuals, test_index, _, _ = deferred_test_points(l2d, data, ROBUSTNESS_INSTANCES, seed)
    deltas, found, _, _ = search_counterfactuals(l2d.model, factuals, threshold, lower, upper, seed)
    base = pairs_frame(l2d.model, factuals, deltas, found).assign(test_index=test_index)

    k, radii = ROBUSTNESS_DIRECTIONS, ROBUSTNESS_RADII
    instance = np.repeat(np.flatnonzero(found), len(radii) * k)
    radius = np.tile(np.repeat(radii, k), len(instance) // (len(radii) * k))
    directions = np.random.default_rng(seed).standard_normal((len(instance), factuals.shape[1]))
    directions *= (radius * deltas[instance].norm(dim=1).numpy() / np.linalg.norm(directions, axis=1))[:, None]
    perturbed = torch.clamp(factuals[instance] + torch.tensor(directions, dtype=torch.float32), lower, upper)
    with torch.no_grad():
        still_deferred = (defer_scores(l2d.model, perturbed) >= threshold).numpy()
    perturbed = perturbed[torch.from_numpy(still_deferred)]
    p_deltas, p_found, _, _ = search_counterfactuals(l2d.model, perturbed, threshold, lower, upper, seed)
    moved = pairs_frame(l2d.model, perturbed, p_deltas, p_found).assign(
        base=instance[still_deferred], radius=radius[still_deferred])

    _write_parquet(path.with_name(path.name + "__perturbed.parquet"), moved)
    save_run(path, base, dict(dataset=dataset, method=method, seed=seed, threshold=threshold,
                              n_perturbed=len(instance), n_still_deferred=int(still_deferred.sum())))
    return path


def load_robustness(dataset: str, method: str, seed: int = 0):
    path = robustness_path(dataset, method, seed)
    base, info = load_run(path)
    moved = pd.read_parquet(path.with_name(path.name + "__perturbed.parquet"), engine="fastparquet")
    return base, moved, info
