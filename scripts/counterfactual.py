import pandas as pd
from joblib import Parallel, delayed

# from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F
import pandas as pd


def evaluate(model, x, threshold_rej=0):
    """
    returns (defer_score, (predicted_class, class_probability))
    """
    # Get original prediction
    with torch.no_grad():
        original_outputs = model(x)
        original_outputs_class = F.softmax(original_outputs[:, :-1], dim=1)
        original_outputs_full = F.softmax(original_outputs, dim=1)
        max_probs, original_predicted_class = torch.max(original_outputs_class, 1)
        original_defer_score = (
            original_outputs_full[0][-1].item()
            - original_outputs_full[0][original_predicted_class[0]].item()
        )
        original_defer = 1 if original_defer_score > threshold_rej else 0
    return original_defer_score, (
        original_predicted_class[0].item(),
        original_outputs_full[0][original_predicted_class[0]].item(),
    )


def find_defer_counterfactual(
    model, x, threshold_rej=0, lambda_l1=10, n_steps=2000, lr=0.1, verbose=False
):
    """
    Finds the perturbation `d` with the minimal L1 norm that changes the deferral decision.
    This version runs for all steps and saves the best solution found.
    """
    model.eval()

    # Get original deferral decision
    original_defer_score, _ = evaluate(model, x, threshold_rej)
    original_defer_decision = 1 if original_defer_score > threshold_rej else 0
    if verbose:
        print(
            f"Original Defer Decision: {'Defer' if original_defer_decision == 1 else 'Do not defer'}. Searching for a counterfactual with minimal L1 distance..."
        )

    d = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([d], lr=lr)

    best_d = None
    min_l1_norm = float("inf")

    for step in range(n_steps):
        optimizer.zero_grad()
        x_cf = x + d
        outputs_cf = model(x_cf)

        # The loss function guides the search
        outputs_full_cf = F.softmax(outputs_cf, dim=1)
        _, predicted_class_cf_idx = torch.max(outputs_full_cf[:, :-1], 1)
        prob_defer_cf = outputs_full_cf[0, -1]
        prob_predicted_class_cf = outputs_full_cf[0, predicted_class_cf_idx[0]]
        defer_score_cf = prob_defer_cf - prob_predicted_class_cf

        loss_cf = (2 * original_defer_decision - 1) * defer_score_cf
        loss_dist = torch.norm(d, p=1)
        loss = loss_cf + lambda_l1 * loss_dist

        loss.backward()
        optimizer.step()

        # --- Check and save the best solution so far ---
        with torch.no_grad():
            # Is the current solution valid?
            current_defer_score, _ = evaluate(model, x_cf, threshold_rej)
            current_defer_decision = 1 if current_defer_score > threshold_rej else 0
            if current_defer_decision != original_defer_decision:
                # Is it better than the best one we've found?
                current_l1_norm = torch.norm(d, p=1).item()
                if current_l1_norm < min_l1_norm:
                    min_l1_norm = current_l1_norm
                    best_d = d.detach().clone()
                    if verbose:
                        print(
                            f"Step {step}: New best solution found! L1 Norm: {min_l1_norm:.4f}"
                        )
    if verbose:
        if best_d is not None:
            print(
                f"Optimization finished. Best counterfactual found with L1 norm: {min_l1_norm:.4f}"
            )
        else:
            print("Could not find a deferral counterfactual within the given steps.")

    return best_d


def process_sample_defer_only(args):
    """
    Worker function for a single sample. Finds and verifies a defer counterfactual.
    """
    # 1. Unpack arguments
    index, sample_x, model, threshold_rej, defer_params = args
    sample_x = sample_x.reshape(1, -1)

    # 2. Get original predictions
    orig_score, (orig_class, _) = evaluate(model, sample_x, threshold_rej)
    orig_defer_decision = 1 if orig_score > threshold_rej else 0

    # 3. Find and VERIFY defer counterfactual
    d_defer = find_defer_counterfactual(model, sample_x, threshold_rej, **defer_params)
    verified_d_defer = None
    if d_defer is not None:
        cf_score, _ = evaluate(model, sample_x + d_defer, threshold_rej)
        cf_defer_decision = 1 if cf_score > threshold_rej else 0
        # Verify that the deferral decision actually flipped
        if cf_defer_decision != orig_defer_decision:
            verified_d_defer = d_defer

    return (index, orig_class, orig_defer_decision, verified_d_defer)


def find_defer_counterfactuals_parallel(
    model, test_x, threshold_rej, defer_params_override=None, n_jobs=-1
):
    """
    Finds defer counterfactuals for an entire dataset in parallel.
    Accepts an override dictionary to customize search parameters.

    Args:
        model (nn.Module): The PyTorch model.
        test_x (torch.Tensor): The dataset of samples to test.
        threshold_rej (float): The rejection threshold.
        defer_params_override (dict, optional): Dictionary of parameters to override
            for find_defer_counterfactual. Defaults to None.
        n_jobs (int): Number of CPU cores to use. -1 means all available.

    Returns:
        tuple: (defer_counterfactuals, main_df)
            - defer_counterfactuals (list): A list of the found counterfactual perturbations (deltas).
            - main_df (pd.DataFrame): A DataFrame summarizing the results for each sample.
    """
    # --- Define Default Parameters ---
    # verbose is False by default for clean parallel execution.
    default_defer_params = {
        "lambda_l1": 0.1,
        "n_steps": 2000,
        "lr": 0.1,
        "verbose": False,
    }

    # --- Create Final Parameters by Applying Overrides ---
    final_defer_params = default_defer_params.copy()
    if defer_params_override:
        final_defer_params.update(defer_params_override)

    # Prepare arguments for each worker process
    tasks = [
        (i, sample, model, threshold_rej, final_defer_params)
        for i, sample in enumerate(test_x)
    ]

    # Run the processing in parallel
    print(f"Processing {len(tasks)} samples for defer counterfactuals in parallel...")
    # Use the new, simpler worker function
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sample_defer_only)(task) for task in tasks
    )

    # --- Assemble the final outputs ---
    defer_counterfactuals = []
    df_rows = []

    for result in results:
        # Unpack the simplified result tuple
        index, orig_class, orig_defer_decision, d_defer = result

        # Determine the index for the counterfactual if it was found
        defer_cf_idx = len(defer_counterfactuals) if d_defer is not None else None
        if d_defer is not None:
            defer_counterfactuals.append(d_defer)

        df_rows.append(
            {
                "sample_index": index,
                "original_class": orig_class,
                "original_defer_decision": orig_defer_decision,
                "defer_cf_index": defer_cf_idx,
            }
        )

    main_df = pd.DataFrame(df_rows)
    # Ensure the index column has a nullable integer type
    main_df["defer_cf_index"] = main_df["defer_cf_index"].astype("Int64")

    return defer_counterfactuals, main_df
