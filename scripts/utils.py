import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


def sigmoid(x):
    """
    Computes the sigmoid function, faithfully replicating the original
    scalar logic with hard boundaries at -15 and 15.

    - Returns exactly 0.0 for x < -15
    - Returns exactly 1.0 for x > 15
    - Computes 1 / (1 + exp(-x)) for x between -15 and 15.

    Args:
        x (float or np.ndarray): The input value(s).

    Returns:
        float or np.ndarray: The sigmoid of the input value(s) with hard boundaries.
    """
    # This is a vectorized equivalent of an if/elif/else statement.
    #
    # 1. Condition: `x > 15`
    #    - If True, the value is 1.0.
    #    - If False, proceed to the next np.where (the "else" part).
    #
    # 2. Nested Condition: `x < -15`
    #    - If True, the value is 0.0.
    #    - If False, calculate the standard sigmoid.

    return np.where(x > 15, 1.0, np.where(x < -15, 0.0, 1 / (1 + np.exp(-x))))


def entropy_expert_err(x, model, expert):
    # Get expert probability
    expert_proba = expert_predict_proba(x, expert)
    x = torch.Tensor(x)
    with torch.no_grad():
        original_outputs = model(x)
        original_outputs_class = F.softmax(original_outputs[:, :-1], dim=1)
        entropy = -torch.sum(
            original_outputs_class * np.log(original_outputs_class + 1e-8), axis=1
        ).numpy()
    expert_err = np.sum(expert_proba * original_outputs_class.detach().numpy(), axis=1)
    return entropy, expert_err


def expert_predict_proba(X_np, expert):
    weights = np.array(expert.w).flatten()
    norm_weights = weights / np.linalg.norm(expert.w)
    projection = expert.alpha * (X_np @ norm_weights)
    prob_fp = sigmoid(expert.fpr_beta + projection)
    prob_fn = sigmoid(expert.fnr_beta - projection)
    return np.c_[prob_fp, prob_fn]


def evaluate(model, x):
    """
    For a single input use .reshape(1,-1)
    """
    with torch.no_grad():
        outputs = model(x)
        outputs_class = F.softmax(outputs[:, :-1], dim=1)
        outputs_full = F.softmax(outputs, dim=1)
        max_probs, predicted_class = torch.max(outputs_class, 1)
        defer_score = (
            outputs_full[:, -1]
            - outputs_full[np.arange(outputs_full.shape[0]), predicted_class]
        )
    return defer_score, outputs_class.detach().numpy(), predicted_class


def build_factual_df(l2d, dataset, expert, defer_only=True):
    factuals = []
    chunks = []
    for test_x, test_y, test_human in dataset.data_test_loader:
        defer_score, class_probs, predicted_class = evaluate(l2d.model, test_x)
        expert_error_probabilities = expert_predict_proba(
            test_x.detach().numpy(), expert
        )
        df_temp = pd.DataFrame(
            {
                # general
                "defer_score": defer_score,
                "label": test_y,
                # model
                "model_pred": predicted_class,
                "entropy": -np.sum(class_probs * np.log(class_probs + 1e-8), axis=1),
                # human
                "expected_expert_err": np.sum(
                    expert_error_probabilities * class_probs, axis=1
                ),
                "expert_pred": test_human,
            }
        )

        # carryover
        factuals.extend(test_x.detach())
        chunks.append(df_temp)
    df = pd.concat(chunks, ignore_index=True)
    if defer_only:
        mask = df["defer_score"] > 0
        return df[mask].reset_index(drop=True), [
            factuals[i] for i in mask.values.nonzero()[0]
        ]
    return df, factuals


def probe_counterfactuals(counter_df, perturbations, factuals, expert, l2d):
    # 2. Now, from this smaller dataframe, drop rows ONLY IF the essential index columns are missing.
    #    This is the key step. We use `subset` to be precise.
    temp = counter_df.dropna(subset=["defer_cf_index"]).copy()
    counterfactuals = np.array(
        [
            factuals[i].detach().numpy() + perturbations[j].flatten().numpy()
            for i, j in zip(temp["sample_index"], temp["defer_cf_index"])
        ]
    )
    defer_score, class_probs, predicted_class = evaluate(
        l2d.model, torch.Tensor(counterfactuals)
    )
    expert_error_probabilities = expert_predict_proba(counterfactuals, expert)

    temp["cfs_entropy"] = -np.sum(class_probs * np.log(class_probs + 1e-8), axis=1)
    temp["cfs_expected_expert_err"] = np.sum(
        expert_error_probabilities * class_probs, axis=1
    )
    return temp
