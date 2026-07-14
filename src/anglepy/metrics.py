'''
This module implements evaluation metrics for circular data.
'''

import torch

def circular_mean_directional_error(y_pred, y_target, is_degrees=False):
    r"""
    Calculates the Circular Mean Directional Error between predicted and target angles.

    This metric evaluates the accuracy of angular predictions by projecting the angular 
    difference into a continuous cosine space. This naturally handles circular wrapping 
    (e.g., the distance between 359 degrees and 1 degree). The resulting error is bounded 
    between 0 (perfect alignment) and 2 (completely opposite directions). 

    The calculation is formally defined as:

    .. math::
        1 - \frac{1}{N} \sum_{i=1}^{N} \cos(y_{\text{target}, i} - y_{\text{pred}, i})

    Args:
        y_pred (torch.Tensor): A tensor of predicted angles.
        y_target (torch.Tensor): A tensor of ground truth target angles. Must be 
            broadcastable to the shape of `y_pred`. It is automatically moved to 
            the same device as `y_pred`.
        is_degrees (bool, optional): If True, indicates that the input tensors are 
            in degrees and automatically converts them to radians before calculation. 
            Defaults to False (assumes inputs are in radians).

    Returns:
        torch.Tensor: A scalar tensor containing the mean circular directional error.

    Example:
        >>> y_pred = torch.tensor([0.0, 3.14159])      # [0, pi]
        >>> y_target = torch.tensor([0.0, 0.0])        # [0, 0]
        >>> # cos(0 - 0) = 1, cos(0 - pi) = -1
        >>> # mean(1, -1) = 0.0 -> 1.0 - 0.0 = 1.0
        >>> circular_mean_directional_error(y_pred, y_target)
        tensor(1.)
        
        >>> # Using degrees
        >>> y_pred_deg = torch.tensor([10.0, 350.0])
        >>> y_target_deg = torch.tensor([10.0, 10.0])  
        >>> circular_mean_directional_error(y_pred_deg, y_target_deg, is_degrees=True)
        tensor(0.0302)
    """
    y_target = y_target.to(y_pred.device)
    # Convert to radians if the inputs are in degrees
    if is_degrees:
        y_pred = torch.deg2rad(y_pred)
        y_target = torch.deg2rad(y_target)
        
    # Calculate the cosine of the angular difference
    cos_diff = torch.cos(y_target - y_pred)
    
    # Calculate and return the mean error
    return 1.0 - torch.mean(cos_diff)

def mean_absolute_angular_deviation(y_pred, y_target, is_degrees=False, return_degrees=True):
    r"""
    Calculates the Mean Absolute Angular Deviation (MAAD) between predicted and target angles.

    This metric computes the shortest angular distance between two angles. It safely 
    handles circular wrapping by projecting the angular difference into Cartesian 
    coordinates (using sine and cosine) and using arctangent to map the error back 
    to the :math:`[-\pi, \pi]` range. It then takes the mean of the absolute deviations.

    Args:
        y_pred (torch.Tensor): A tensor of predicted angles.
        y_target (torch.Tensor): A tensor of ground truth target angles. Must be 
            broadcastable to the shape of `y_pred`. It is automatically moved to 
            the same device as `y_pred`.
        is_degrees (bool, optional): If True, indicates that the input tensors 
            (`y_pred` and `y_target`) are in degrees. Defaults to False (radians).
        return_degrees (bool, optional): If True, the final computed mean deviation 
            is returned in degrees. If False, it is returned in radians. Defaults to True.

    Returns:
        torch.Tensor: A scalar tensor containing the mean absolute angular deviation.

    Example:
        >>> # The naive difference between 350° and 10° is 340°.
        >>> # The shortest angular path is actually 20°.
        >>> y_pred = torch.tensor([350.0])
        >>> y_target = torch.tensor([10.0])
        >>> mean_absolute_angular_deviation(y_pred, y_target, is_degrees=True, return_degrees=True)
        tensor(20.)
        
        >>> # Using radians, returning radians
        >>> y_pred_rad = torch.tensor([1.5 * 3.14159])  # ~270 degrees
        >>> y_target_rad = torch.tensor([0.0])          # 0 degrees
        >>> mean_absolute_angular_deviation(y_pred_rad, y_target_rad, is_degrees=False, return_degrees=False)
        tensor(1.5708)  # ~pi/2 radians (90 degrees)
    """
    y_target = y_target.to(y_pred.device)

    # Calculate the raw difference
    diff = y_target - y_pred
    
    # Convert to radians for the trigonometric functions if needed
    if is_degrees:
        diff_rad = torch.deg2rad(diff)
    else:
        diff_rad = diff
        
    # Wrap the difference to the shortest path [-pi, pi]
    wrapped_diff_rad = torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))
    
    # Take the absolute value
    abs_dev_rad = torch.abs(wrapped_diff_rad)
    
    # Convert back to degrees if requested, then compute the mean
    if return_degrees:
        abs_dev = torch.rad2deg(abs_dev_rad)
    else:
        abs_dev = abs_dev_rad
        
    return torch.mean(abs_dev)

def angular_distance(a, b):
    """
    Calculates the shortest angular distance between two angles in radians.
    """
    diff = torch.abs(a - b) % (2 * torch.pi)
    return torch.minimum(diff, 2 * torch.pi - diff)

def circular_crps(y_pred_ensemble, y_true):
    r"""
    Calculates the Circular Continuous Ranked Probability Score (CRPS) for an ensemble of predictions.

    CRPS is a strictly proper scoring rule used to evaluate both the accuracy and 
    calibration of probabilistic forecasts. This implementation adapts standard CRPS 
    for circular or angular variables by using the shortest angular distance between 
    points instead of the standard absolute difference (Gneiting, 2007). 

    The empirical circular CRPS is calculated as:

    .. math::
        CRPS = \frac{1}{M} \sum_{i=1}^{M} d(\hat{y}_i, y) - \frac{1}{2 M^2} \sum_{i=1}^{M} \sum_{j=1}^{M} d(\hat{y}_i, \hat{y}_j)

    where :math:`y` is the true observation, :math:`\hat{y}_i` is an ensemble member, :math:`M` is the 
    total number of ensemble members, and :math:`d(\cdot, \cdot)` is the angular distance.

    Args:
        y_pred_ensemble (torch.Tensor or array-like): The ensemble of predicted angles 
            in radians. Expected shapes are :math:`(N, M)` for univariate predictions or 
            :math:`(N, D, M)` for multivariate predictions, where :math:`N` is the number of observations, 
            :math:`D` is the number of feature dimensions, and :math:`M` is the number of ensemble members.
        y_true (torch.Tensor or array-like): The ground truth angles in radians. 
            Expected shapes are :math:`(N,)` for univariate targets or :math:`(N, D)` for multivariate 
            targets. It is automatically converted to a tensor and moved to the same 
            device/dtype as `y_pred_ensemble`.

    Returns:
        torch.Tensor: A tensor containing the computed circular CRPS values. The shape 
        exactly matches the shape of the input `y_true` (:math:`(N,)` or :math:`(N, D)`).

    Raises:
        ValueError: If the number of observations (:math:`N`) or the feature dimension size (:math:`D`) of 
            `y_true` does not match the corresponding dimensions of `y_pred_ensemble`.

    Example:
        >>> # Univariate example: #Observations N=2, Ensemble size M=3
        >>> y_true = torch.tensor([0.0, 3.14159])                  # True angles: 0, pi
        >>> y_pred = torch.tensor([[0.1, -0.1, 0.0],               # Predictions close to 0
        ...                        [3.0, -3.0, 3.14159]])          # Predictions close to pi
        >>> crps = circular_crps(y_pred, y_true)
        >>> crps.shape
        torch.Size([2])
        
        >>> # Multivariate example: #Observations N=2, Features D=2, Ensemble size M=5
        >>> y_true_mv = torch.zeros(2, 2)
        >>> y_pred_mv = torch.zeros(2, 2, 5)
        >>> crps_mv = circular_crps(y_pred_mv, y_true_mv)
        >>> crps_mv.shape
        torch.Size([2, 2])
    """
    # Convert to tensors if they aren't already
    y_pred = torch.as_tensor(y_pred_ensemble)
    
    # Ensure y_true matches the device and dtype of y_pred
    y_true = torch.as_tensor(y_true, dtype=y_pred.dtype, device=y_pred.device)
    
    # Track the original number of dimensions to format the output correctly
    original_ndim_true = y_true.ndim
    
    # Standardize Inputs to 3D: (N, D, M)
    # If no response_dim is present, add it as a dimension of size 1
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(1)  # Transforms (N,) to (N, 1)
        
    if y_pred.ndim == 2:
        y_pred = y_pred.unsqueeze(1)  # Transforms (N, M) to (N, 1, M)

    # Sanity check to ensure broadcasting will work
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Data size mismatch: y_true has {y_true.shape[0]}, y_pred has {y_pred.shape[0]}")
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(f"Response dim mismatch: y_true has {y_true.shape[1]}, y_pred has {y_pred.shape[1]}")

    # Compute Term 1: Distance between ensemble and true observation
    # Expand y_true to (N, D, 1) so it broadcasts against (N, D, M)
    y_true_expanded = y_true.unsqueeze(-1)
    dist_obs = angular_distance(y_pred, y_true_expanded)
    
    # Mean over the ensemble members (dim=-1)
    term1 = torch.mean(dist_obs, dim=-1)  # Shape: (N, D)
    
    # Compute Term 2: Pairwise distance among ensemble members
    # Expand y_pred to (N, D, M, 1) and (N, D, 1, M) to get the M x M combinations
    pred_m = y_pred.unsqueeze(-1)
    pred_n = y_pred.unsqueeze(-2)
    
    dist_ens = angular_distance(pred_m, pred_n)
    
    # Mean over both ensemble axes (dim=-1 and dim=-2), multiplied by 0.5 per the formula
    term2 = 0.5 * torch.mean(dist_ens, dim=(-2, -1))  # Shape: (N, D)
    
    # Final CRPS calculation
    crps = term1 - term2  # Shape: (N, D)
    
    # Restore Original Shape Format
    # If the user passed 1D data (N,), return 1D CRPS (N,)
    if original_ndim_true == 1:
        crps = torch.squeeze(crps, dim=-1)
        
    return crps

def mean_circular_crps(y_pred_ensemble, y_true, return_degrees=True):
    r"""
    Computes the scalar mean circular CRPS over all observations and response dimensions.

    This function acts as a wrapper around `circular_crps`. It calculates the individual 
    Continuous Ranked Probability Score (CRPS) for each observation and dimension, aggregates 
    them into a single scalar mean, and optionally converts the results from radians to degrees.

    Args:
        y_pred_ensemble (torch.Tensor or array-like): The ensemble of predicted angles 
            in radians. Expected shapes are :math:`(N, M)` for univariate or :math:`(N, D, M)` for 
            multivariate predictions.
        y_true (torch.Tensor or array-like): The ground truth angles in radians. 
            Expected shapes are :math:`(N,)` or :math:`(N, D)`.
        return_degrees (bool, optional): If True, converts both the final mean scalar 
            and the raw CRPS tensor from radians to degrees before returning. 
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - mean_crps (float): The aggregated mean circular CRPS across all 
              observations and dimensions, returned as a standard Python float.
            - crps_tensor (torch.Tensor): The raw tensor of CRPS scores corresponding 
              to each observation. Shape matches `y_true`.

    Example:
        >>> # #Observations N=2, Ensemble size M=3
        >>> y_true = torch.tensor([0.0, 3.14159])                  # [0, pi]
        >>> y_pred = torch.tensor([[0.1, -0.1, 0.0],               # Predictions close to 0
        ...                        [3.0, -3.0, 3.14159]])          # Predictions close to pi
        >>> mean_score, raw_scores = mean_circular_crps(y_pred, y_true, return_degrees=True)
        >>> type(mean_score)
        <class 'float'>
        >>> raw_scores.shape
        torch.Size([2])
    """
    # Get the raw tensor of scores
    crps_tensor = circular_crps(y_pred_ensemble, y_true)
    
    # Compute the scalar mean across all dimensions
    mean_crps = torch.mean(crps_tensor)
    
    # Return standard float for the mean score, but keep the raw scores as a tensor
    if return_degrees:
        return torch.rad2deg(torch.tensor(mean_crps.detach().clone())).item(), torch.rad2deg(crps_tensor)
    return mean_crps.item(), crps_tensor

def accuracy(y_pred, y_test, theta=torch.pi/6):
    r"""
    Computes the threshold-based accuracy for angular predictions.

    This metric evaluates the proportion of predictions that fall within a specified 
    angular tolerance (`theta`) of the ground truth. It safely handles circular 
    wrapping by calculating the shortest angular path (e.g., the distance between 
    :math:`359^\circ` and :math:`1^\circ` is :math:`2^\circ`, not :math:`358^\circ`) before evaluating against 
    the threshold.

    Args:
        y_pred (torch.Tensor): A tensor of predicted angles in radians.
        y_test (torch.Tensor): A tensor of ground truth angles in radians. Must be 
            broadcastable to the shape of `y_pred`.
        theta (float, optional): The maximum allowed angular deviation (in radians) 
            for a prediction to be considered "correct". Defaults to :math:`\pi/6` (30 degrees).

    Returns:
        float: The fraction of predictions that are within the `theta` threshold. 
        Returns a value between 0.0 (no correct predictions) and 1.0 (all correct).

    Example:
        >>> # pi/6 is approx 0.523 radians (30 degrees)
        >>> y_pred = torch.tensor([0.0, 3.14159, 6.28])     # [0, pi, ~2*pi]
        >>> y_test = torch.tensor([0.2, 0.0, 0.0])          # [0.2, 0, 0]
        >>> # Distances:
        >>> # 1: |0.0 - 0.2| = 0.2 rad (<= 0.523) -> Correct
        >>> # 2: |pi - 0.0| = 3.14 rad (> 0.523) -> Incorrect
        >>> # 3: |~2*pi - 0| = ~0.0 rad (<= 0.523) -> Correct (due to wrapping)
        >>> accuracy(y_pred, y_test, theta=torch.pi/6)
    """
    y_pred = y_pred.to(y_test.device)
    # Calculate the shortest angular distance between predicted and true angle
    diff = y_pred - y_test
    angular_distances = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()

    # Fraction of predictions where the distance is strictly less than or equal to theta
    correct_mask = angular_distances <= theta
    acc_theta = torch.mean(correct_mask.float()).item()
    return acc_theta
  
    
def med_err(y_pred, y_test, return_degrees=True):
    r"""
    Calculates the median absolute angular error between predicted and target angles.

    This metric evaluates the central tendency of the prediction errors. It safely 
    handles circular wrapping by computing the shortest angular path (e.g., the 
    distance between :math:`2\pi` and :math:`0.1` radians is :math:`0.1` radians) before finding the median 
    across the entire batch. Using the median makes this metric robust to outliers 
    compared to mean-based metrics.

    Args:
        y_pred (torch.Tensor): A tensor of predicted angles in radians.
        y_test (torch.Tensor): A tensor of ground truth angles in radians. Must be 
            broadcastable to the shape of `y_pred`. It is automatically moved to 
            the same device as `y_pred`.
        return_degrees (bool, optional): If True, converts the final computed median 
            error from radians to degrees. If False, returns the error in radians. 
            Defaults to True.

    Returns:
        float: The median absolute angular error as a Python float.

    Example:
        >>> y_pred = torch.tensor([0.0, 3.14159, 6.28])     # [0, pi, ~2*pi]
        >>> y_test = torch.tensor([0.1, 3.14159, 0.0])      # [0.1, pi, 0]
        >>> med_err(y_pred, y_test, return_degrees=True)
    """
    y_pred = y_pred.to(y_test.device)
    
    # Calculate the shortest angular distance between predicted and true azimuth
    diff = y_pred - y_test
    angular_distances = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()

    # The metric computes the median of these distances.
    med_err_rad = torch.median(angular_distances)

    # Return as standard float, applying unit conversion if requested
    if return_degrees:
        return med_err_rad.item() * (180.0 / torch.pi)
    
    return med_err_rad.item()
