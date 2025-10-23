#!/usr/bin/env python3
"""
Uncertainty quantification and calibration for ClinFuseDiff++
Implements ECE (Expected Calibration Error) and Brier score for spatial uncertainty
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def compute_ensemble_uncertainty(
    samples: torch.Tensor,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ensemble-based uncertainty from multiple diffusion samples

    Args:
        samples: Tensor of shape (N, C, D, H, W) where N is number of samples
        reduction: How to reduce across channels ('mean', 'max', None)

    Returns:
        mean: Mean prediction across samples (C, D, H, W)
        variance: Variance map (C, D, H, W) or (D, H, W) if reduced
    """
    # Compute mean and variance across samples
    mean = samples.mean(dim=0)  # (C, D, H, W)
    variance = samples.var(dim=0, unbiased=True)  # (C, D, H, W)

    if reduction == "mean":
        variance = variance.mean(dim=0)  # (D, H, W)
    elif reduction == "max":
        variance = variance.max(dim=0)[0]  # (D, H, W)

    return mean, variance


def compute_predictive_entropy(
    samples: torch.Tensor,
    bins: int = 256,
    epsilon: float = 1e-10
) -> torch.Tensor:
    """
    Compute predictive entropy from ensemble samples

    Args:
        samples: (N, C, D, H, W) ensemble samples
        bins: Number of bins for histogram
        epsilon: Small constant for numerical stability

    Returns:
        entropy: (D, H, W) entropy map
    """
    N = samples.shape[0]

    # Compute mean prediction
    mean_pred = samples.mean(dim=0)  # (C, D, H, W)

    # For each spatial location, compute entropy
    # H(p) = -sum(p * log(p))

    # Simple version: treat intensity as probability (after normalization)
    # Normalize to [0, 1]
    min_val = mean_pred.min()
    max_val = mean_pred.max()
    normalized = (mean_pred - min_val) / (max_val - min_val + epsilon)

    # Compute entropy per channel then average
    entropy = -normalized * torch.log(normalized + epsilon)
    entropy = entropy.mean(dim=0)  # Average across channels

    return entropy


def expected_calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) for regression/reconstruction

    Calibration: Does predicted uncertainty match actual error?

    Args:
        predictions: Predicted values (D, H, W) or (N, D, H, W)
        targets: Ground truth values (D, H, W) or (N, D, H, W)
        uncertainties: Predicted uncertainties (D, H, W) or (N, D, H, W)
        n_bins: Number of calibration bins
        mask: Optional binary mask to compute ECE only in ROI

    Returns:
        dict with:
            - ece: Expected Calibration Error
            - bin_errors: Per-bin absolute errors
            - bin_confs: Per-bin mean confidence (inverse uncertainty)
            - bin_accs: Per-bin mean accuracy (negative error)
            - bin_counts: Number of samples per bin
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    uncertainties = np.asarray(uncertainties).flatten()

    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        predictions = predictions[mask]
        targets = targets[mask]
        uncertainties = uncertainties[mask]

    # Compute per-voxel errors
    errors = np.abs(predictions - targets)

    # Convert uncertainty to confidence (inverse relationship)
    # Normalize uncertainties to [0, 1] range
    unc_min, unc_max = uncertainties.min(), uncertainties.max()
    if unc_max > unc_min:
        normalized_unc = (uncertainties - unc_min) / (unc_max - unc_min)
    else:
        normalized_unc = np.zeros_like(uncertainties)

    confidences = 1.0 - normalized_unc  # Higher confidence = lower uncertainty

    # Create bins based on confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute per-bin statistics
    bin_errors = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask_i = bin_indices == i
        if mask_i.sum() > 0:
            bin_errors[i] = errors[mask_i].mean()
            bin_confs[i] = confidences[mask_i].mean()
            bin_counts[i] = mask_i.sum()

    # ECE: weighted average of |confidence - accuracy|
    # For regression: accuracy = 1 - normalized_error
    total_samples = bin_counts.sum()
    if total_samples > 0:
        # Normalize errors to [0, 1] for fair comparison with confidence
        max_error = errors.max()
        if max_error > 0:
            normalized_errors = errors / max_error

            # Recalculate bin errors with normalization
            for i in range(n_bins):
                mask_i = bin_indices == i
                if mask_i.sum() > 0:
                    bin_errors[i] = normalized_errors[mask_i].mean()

            bin_accs = 1.0 - bin_errors  # Accuracy = 1 - error

            # ECE = sum over bins of: (count_i / total) * |conf_i - acc_i|
            ece = np.sum((bin_counts / total_samples) * np.abs(bin_confs - bin_accs))
        else:
            ece = 0.0
            bin_accs = np.ones(n_bins)
    else:
        ece = 0.0
        bin_accs = np.zeros(n_bins)

    return {
        "ece": float(ece),
        "bin_errors": bin_errors.tolist(),
        "bin_confs": bin_confs.tolist(),
        "bin_accs": bin_accs.tolist(),
        "bin_counts": bin_counts.tolist()
    }


def brier_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalize: bool = True
) -> float:
    """
    Compute Brier score for probabilistic predictions

    Brier = mean((prediction - target)^2 + uncertainty_penalty)

    Lower is better. Penalizes both inaccurate predictions and miscalibrated uncertainty.

    Args:
        predictions: Predicted values (D, H, W)
        targets: Ground truth values (D, H, W)
        uncertainties: Predicted uncertainties (variance) (D, H, W)
        mask: Optional binary mask
        normalize: Normalize values to [0, 1] before computing score

    Returns:
        brier_score: Scalar brier score
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    uncertainties = np.asarray(uncertainties).flatten()

    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        predictions = predictions[mask]
        targets = targets[mask]
        uncertainties = uncertainties[mask]

    if normalize:
        # Normalize to [0, 1]
        min_val = min(predictions.min(), targets.min())
        max_val = max(predictions.max(), targets.max())
        if max_val > min_val:
            predictions = (predictions - min_val) / (max_val - min_val)
            targets = (targets - min_val) / (max_val - min_val)
            uncertainties = uncertainties / ((max_val - min_val) ** 2)

    # Brier score: MSE + uncertainty calibration term
    mse = np.mean((predictions - targets) ** 2)

    # Uncertainty should match squared error
    squared_errors = (predictions - targets) ** 2
    uncertainty_calibration = np.mean((uncertainties - squared_errors) ** 2)

    # Combined Brier score
    brier = mse + 0.5 * uncertainty_calibration

    return float(brier)


def reliability_diagram_data(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
    mask: Optional[np.ndarray] = None
) -> Dict:
    """
    Prepare data for reliability diagram (calibration plot)

    Returns bin statistics for plotting confidence vs accuracy

    Args:
        predictions: Predicted values
        targets: Ground truth values
        uncertainties: Predicted uncertainties
        n_bins: Number of bins
        mask: Optional ROI mask

    Returns:
        dict with bin_confs, bin_accs, bin_counts for plotting
    """
    ece_result = expected_calibration_error(
        predictions, targets, uncertainties, n_bins, mask
    )

    return {
        "bin_confidences": ece_result["bin_confs"],
        "bin_accuracies": ece_result["bin_accs"],
        "bin_counts": ece_result["bin_counts"],
        "ece": ece_result["ece"]
    }


def temperature_scaling_calibration(
    uncertainties: torch.Tensor,
    errors: torch.Tensor,
    initial_temp: float = 1.0,
    lr: float = 0.01,
    max_iters: int = 50
) -> Tuple[float, torch.Tensor]:
    """
    Learn temperature scaling parameter to calibrate uncertainties

    Optimizes T such that uncertainties/T better match actual errors

    Args:
        uncertainties: Predicted uncertainties (N,)
        errors: Actual errors (N,)
        initial_temp: Initial temperature
        lr: Learning rate
        max_iters: Maximum optimization iterations

    Returns:
        temperature: Learned temperature scalar
        calibrated_uncertainties: uncertainties / temperature
    """
    temperature = torch.tensor([initial_temp], requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iters)

    uncertainties = uncertainties.detach()
    errors = errors.detach()

    def closure():
        optimizer.zero_grad()
        calibrated_unc = uncertainties / temperature
        # Loss: MSE between calibrated uncertainty and actual error
        loss = F.mse_loss(calibrated_unc, errors)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature_value = temperature.item()
    calibrated_uncertainties = uncertainties / temperature_value

    return temperature_value, calibrated_uncertainties


def compute_calibration_metrics(
    ensemble_samples: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_bins: int = 10
) -> Dict:
    """
    Compute comprehensive calibration metrics from ensemble samples

    Args:
        ensemble_samples: (N, D, H, W) ensemble predictions
        ground_truth: (D, H, W) ground truth
        mask: Optional (D, H, W) binary mask
        n_bins: Number of calibration bins

    Returns:
        dict with ece, brier, reliability_data
    """
    # Compute mean prediction and uncertainty
    mean_pred = ensemble_samples.mean(axis=0)  # (D, H, W)
    variance = ensemble_samples.var(axis=0, ddof=1)  # (D, H, W)

    # Compute metrics
    ece_result = expected_calibration_error(
        mean_pred, ground_truth, variance, n_bins, mask
    )

    brier = brier_score(
        mean_pred, ground_truth, variance, mask, normalize=True
    )

    reliability_data = reliability_diagram_data(
        mean_pred, ground_truth, variance, n_bins, mask
    )

    return {
        "ece": ece_result["ece"],
        "brier": brier,
        "reliability_diagram": reliability_data,
        "mean_uncertainty": float(variance.mean()) if mask is None else float(variance[mask > 0].mean()),
        "mean_error": float(np.abs(mean_pred - ground_truth).mean()) if mask is None else float(np.abs(mean_pred - ground_truth)[mask > 0].mean())
    }