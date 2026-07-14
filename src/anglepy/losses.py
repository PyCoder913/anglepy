'''
This module implements the energy score loss function using geodesic, chordal, or cosine distance for angular data.
'''

import torch
from .utils import vectorize
from .kernels import *


def chordal_dist(t1, t2, dim=-1):
    r"""
    Calculates the chordal distance between two tensors of angles.

    The chordal distance is the straight-line Euclidean distance between two angles 
    when they are embedded as points on a unit circle. Unlike the arc length (angular 
    distance), the chordal distance uses Cartesian coordinates, making it a continuous 
    and differentiable metric that naturally avoids wrapping discontinuities at :math:`2\pi`.

    For vectors of angles, this function computes the :math:`L_2` norm of the element-wise 
    Cartesian differences along the specified dimension:

    .. math::
        \sqrt{\sum_{i} \left( (\cos(t_{1,i}) - \cos(t_{2,i}))^2 + (\sin(t_{1,i}) - \sin(t_{2,i}))^2 \right) + \epsilon}
    
    Note:
        A small epsilon (:math:`1 \times 10^{-8}`) is added inside the square root to 
        prevent infinite gradients (NaNs) during backpropagation when the distance 
        is exactly zero.

    Args:
        t1 (torch.Tensor): The first tensor of angles in radians.
        t2 (torch.Tensor): The second tensor of angles in radians. Must be 
            broadcastable to the shape of `t1`.
        dim (int, optional): The dimension over which to sum the squared 
            Cartesian differences before applying the square root. Defaults to -1 
            (the last dimension).

    Returns:
        torch.Tensor: A tensor containing the calculated chordal distances. 
        The dimension specified by `dim` is reduced.
    """
    # Calculate Cartesian differences
    cos_diff = torch.cos(t1) - torch.cos(t2)
    sin_diff = torch.sin(t1) - torch.sin(t2)
    
    # Squared element-wise distance: (cos(t1) - cos(t2))^2 + (sin(t1) - sin(t2))^2
    elementwise_sq_dist = cos_diff**2 + sin_diff**2
    
    # Sum over the data dimension and take the square root (L2 norm)
    # Adding a tiny epsilon (1e-8) before sqrt prevents NaN gradients if the distance is exactly 0
    return torch.sqrt(torch.sum(elementwise_sq_dist, dim=dim) + 1e-8)

def geodesic_dist(t1, t2, dim=-1):
    r"""
    Calculates the geodesic (shortest arc) distance between two tensors of angles.

    Unlike the chordal distance which measures the straight line through the circle, 
    the geodesic distance measures the length of the shortest path along the circle's 
    circumference. It safely handles angular wrapping by computing the minimum of the 
    clockwise and counter-clockwise arc lengths between the angles. 

    For vectors of angles, it computes the :math:`L_2` norm of these element-wise shortest arcs 
    along the specified dimension:

    .. math::
        \sqrt{\sum_{i} \left( \min(\delta_i, 2\pi - \delta_i) \right)^2}

    where :math:`\delta_i = |t_{1,i} - t_{2,i}| \bmod 2\pi`.

    Args:
        t1 (torch.Tensor): The first tensor of angles in radians.
        t2 (torch.Tensor): The second tensor of angles in radians. Must be 
            broadcastable to the shape of `t1`.
        dim (int, optional): The dimension over which to compute the :math:`L_2` vector norm 
            of the arc lengths. Defaults to -1 (the last dimension).

    Returns:
        torch.Tensor: A tensor containing the computed geodesic distances. 
        The dimension specified by `dim` is reduced.
    """
    diff = torch.abs(t1 - t2) % (2 * torch.pi) # Absolute difference modulo 2pi
    arc = torch.min(diff, 2 * torch.pi - diff) # Shortest arc length (min of clockwise/counter-clockwise) 
    return torch.norm(arc, p=2, dim=dim) # Vector norm over the Data Dimension (last dim)

def cosine_dist(t1, t2, dim=-1):
    r"""
    Calculates the mean cosine-based circular distance between two tensors of angles.

    This metric computes the Circular Mean Directional Error by evaluating 
    :math:`1 - \cos(t_1 - t_2)` element-wise. This formulation natively 
    handles circular wrapping boundaries (e.g., the difference between :math:`2\pi` 
    and :math:`0`). The resulting element-wise distances range from :math:`0` (perfectly 
    aligned) to :math:`2` (completely opposite), which are then averaged across the 
    specified dimension.

    Args:
        t1 (torch.Tensor): The first tensor of angles in radians.
        t2 (torch.Tensor): The second tensor of angles in radians. Must be 
            broadcastable to the shape of `t1`.
        dim (int, optional): The dimension over which to compute the mean of 
            the element-wise cosine distances. Defaults to -1 (the last dimension).

    Returns:
        torch.Tensor: A tensor containing the computed mean cosine distances. 
        The dimension specified by `dim` is reduced.
    """
    # Calculate 1 - cos(diff) for each element
    elementwise_cos_dist = 1.0 - torch.cos(t1 - t2)
    
    # Mean over the Data Dimension (last dim)
    return torch.mean(elementwise_cos_dist, dim=dim)


def energy_loss(x_true, x_est, gamma=1, verbose=True, dist_method="chordal"):
    r"""
    Computes the energy score loss for probabilistic predictions of circular data.

    The Energy Score is a strictly proper scoring rule that generalizes the Continuous 
    Ranked Probability Score (CRPS) to multivariate settings. This implementation adapts 
    the metric for angular data (in radians) by utilizing circular distance functions 
    ("chordal" or "cosine"). 

    The empirical Energy Loss is computed as:

    .. math::
        \mathbb{E}[d(\hat{x}, x)^\gamma] - \frac{1}{2} \mathbb{E}[d(\hat{x}_i, \hat{x}_j)^\gamma]

    where :math:`x` is the ground truth, :math:`\hat{x}` are the ensemble predictions, :math:`m` is the 
    number of ensemble members, and :math:`d(\cdot, \cdot)` is the specified circular distance. 

    Args:
        x_true (torch.Tensor): The ground truth angles in radians. Expected to be a tensor 
            that can be reshaped to `(Batch, Dim)`.
        x_est (torch.Tensor or list of torch.Tensor): The ensemble of predicted angles in radians. 
            If a tensor, it is split across the batch dimension. If a list, it is stacked 
            to form a shape of `(Batch, Samples, Dim)`.
        gamma (float, optional): The power parameter applied to the distances. If `gamma` 
            is not an integer, a small epsilon is added to the distances before exponentiation 
            to prevent NaN gradients. Defaults to 1.
        verbose (bool, optional): If True, returns a concatenated tensor containing the 
            total loss, the distance to ground truth (Term 1), and the pairwise ensemble 
            distance (Term 2). If False, returns only the total scalar loss. Defaults to True.
        dist_method (str, optional): The distance metric to evaluate. Must be either 
            `"chordal"` or `"cosine"`. Defaults to `"chordal"`.

    Returns:
        torch.Tensor: 
        - If `verbose` is True, returns a 1D tensor of shape `(3,)` containing: 
          `[total_energy_loss, term_1, term_2]`.
        - If `verbose` is False, returns a scalar tensor containing the computed 
          total energy loss.

    Raises:
        ValueError: If `dist_method` is not one of the supported methods.
        
    Note:
        This function relies on an external `vectorize` function and circular distance 
        functions (`chordal_dist`, `cosine_dist`) which must be available in the local scope.
    """
    EPS = 0 if float(gamma).is_integer() else 1e-5
    
    # Input Processing
    x_true = vectorize(x_true).unsqueeze(1) # Shape: [Batch, 1, Dim]
    
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))
    m = len(x_est)
    x_est = [vectorize(x_est[i]).unsqueeze(1) for i in range(m)]
    x_est = torch.cat(x_est, dim=1) # Shape: [Batch, Samples, Dim]
    
    # Term 1: Distance between Estimated Samples and Ground Truth
    # x_est: [B, m, D] | x_true: [B, 1, D] -> Broadcasting works automatically
    if dist_method == "chordal":
        d_xt = chordal_dist(x_est, x_true) # Result: [B, m]
    elif dist_method == "cosine":
        d_xt = cosine_dist(x_est, x_true) # Result: [B, m]
    else:
        raise ValueError(f"dist_method can be one of `chordal` or `cosine`.")
    s1 = (d_xt + EPS).pow(gamma).mean()

    # Term 2: Pairwise distances within Estimated Samples
    # We broadcast to create an [m, m] matrix for every batch
    # x_est_a: [B, m, 1, D]
    x_est_a = x_est.unsqueeze(2)
    # x_est_b: [B, 1, m, D]
    x_est_b = x_est.unsqueeze(1)
    
    # Result: [B, m, m]
    if dist_method == "chordal":
        d_xx = chordal_dist(x_est_a, x_est_b)
    elif dist_method == "cosine":
        d_xx = cosine_dist(x_est_a, x_est_b)
    else:
        raise ValueError(f"dist_method can be one of `geodesic`, `chordal`, or `cosine`.")
    
    # Bias correction factor for energy score
    s2 = (d_xx + EPS).pow(gamma).mean() * m / (m - 1)

    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2 / 2)
    

def energy_loss_two_sample(x0, x, xp, x0p=None, gamma=1, verbose=True, weights=None, dist_method="chordal"):
    r"""
    Computes a two-sample Energy Loss (or Energy Distance) for circular data.

    This function evaluates the divergence between an estimated distribution and a 
    target distribution using discrete samples. It operates in two modes:
    
    1. **Standard Energy Score (`x0p` is None)**: Uses one sample from the ground 
       truth (:math:`x_0`) and two independent samples from the estimated distribution (:math:`x, x'`). 
       Evaluates the strictly proper scoring rule:

       .. math::
           \frac{1}{2} \left( \mathbb{E}[d(x, x_0)^\gamma] + \mathbb{E}[d(x', x_0)^\gamma] \right) - \frac{1}{2} \mathbb{E}[d(x, x')^\gamma]
       
    2. **Maximum Mean Discrepancy / Energy Distance (`x0p` is provided)**: Uses two 
       samples from the true distribution (:math:`x_0, x_0'`) and two from the estimated 
       distribution (:math:`x, x'`). Evaluates the full symmetric energy distance:

       .. math::
           \mathbb{E}[d(x, x_0)^\gamma] - \frac{1}{2} \mathbb{E}[d(x, x')^\gamma] - \frac{1}{2} \mathbb{E}[d(x_0, x_0')^\gamma]

    Distances are computed using specified circular metrics to correctly handle 
    angular wrapping boundaries. All inputs are expected to be in radians.

    Args:
        x0 (torch.Tensor): First sample from the ground truth distribution.
        x (torch.Tensor): First sample from the estimated (predicted) distribution.
        xp (torch.Tensor): Second independent sample from the estimated distribution.
        x0p (torch.Tensor, optional): Second independent sample from the ground truth 
            distribution. If provided, calculates the full two-sample energy distance. 
            Defaults to None.
        gamma (float, optional): Power parameter applied to the distances. If `gamma` 
            is not an integer, a small epsilon is added to distances to prevent 
            NaN gradients. Defaults to 1.
        verbose (bool, optional): If True, returns a concatenated tensor with the 
            total loss, the cross-term distance (Term 1), and the internal estimated 
            distance (Term 2). If False, returns only the total scalar loss. Defaults to True.
        weights (torch.Tensor, optional): Sample weights applied when computing the 
            loss. Only used if `x0p` is None. Defaults to uniform weights (`1 / N`).
        dist_method (str, optional): The circular distance metric to use. Must be 
            either `"chordal"` or `"cosine"`. Defaults to `"chordal"`.

    Returns:
        torch.Tensor: 
            - If `verbose=True`: A 1D tensor of shape `(3,)` containing 
              `[total_loss, term_1, term_2]`.
            - If `verbose=False`: A scalar tensor containing the computed loss.

    Raises:
        ValueError: If `dist_method` is not one of the supported strings.

    Note:
        This function relies on an external `vectorize` function and circular distance 
        functions (`chordal_dist`, `cosine_dist`) which must be available in the local scope.
    """
    EPS = 0 if float(gamma).is_integer() else 1e-5
    
    # Ensure inputs are tensors
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)

    if weights is None:
        weights = 1 / x0.size(0)

    if x0p is None:
        # Cross terms (True vs Est)
        if dist_method == "chordal":
            d_x_x0 = chordal_dist(x, x0, dim=1)
            d_xp_x0 = chordal_dist(xp, x0, dim=1)
        elif dist_method == "cosine":
            d_x_x0 = cosine_dist(x, x0, dim=1)
            d_xp_x0 = cosine_dist(xp, x0, dim=1)
        
        else:
            raise ValueError(f"dist_method can be one of geodesic, chordal, or cosine.")
        s1 = ((d_x_x0 + EPS).pow(gamma) * weights).sum() / 2 + \
             ((d_xp_x0 + EPS).pow(gamma) * weights).sum() / 2
        
        # Internal term (Est vs Est)
        if dist_method == "chordal":
            d_x_xp = chordal_dist(x, xp, dim=1)
        elif dist_method == "cosine":
            d_x_xp = cosine_dist(x, xp, dim=1)
        
        else:
            raise ValueError(f"dist_method can be one of geodesic, chordal, or cosine.")
        s2 = ((d_x_xp + EPS).pow(gamma) * weights).sum()
        
        loss = s1 - s2/2
        
    else:
        x0p = vectorize(x0p)
        
        # Cross terms (True vs Est)
        if dist_method == "chordal":
            c1 = (chordal_dist(x, x0, dim=1) + EPS).pow(gamma).sum()
            c2 = (chordal_dist(xp, x0, dim=1) + EPS).pow(gamma).sum()
            c3 = (chordal_dist(x, x0p, dim=1) + EPS).pow(gamma).sum()
            c4 = (chordal_dist(xp, x0p, dim=1) + EPS).pow(gamma).sum()
        elif dist_method == "cosine":
            c1 = (cosine_dist(x, x0, dim=1) + EPS).pow(gamma).sum()
            c2 = (cosine_dist(xp, x0, dim=1) + EPS).pow(gamma).sum()
            c3 = (cosine_dist(x, x0p, dim=1) + EPS).pow(gamma).sum()
            c4 = (cosine_dist(xp, x0p, dim=1) + EPS).pow(gamma).sum()
        else:
            raise ValueError(f"dist_method can be one of geodesic, chordal, or cosine.")
        
        s1 = (c1 + c2 + c3 + c4) / 4
    
        if dist_method == "chordal":
            # Internal term (Est vs Est)
            s2 = (chordal_dist(x, xp, dim=1) + EPS).pow(gamma).sum()
            # Internal term (True vs True)
            s3 = (chordal_dist(x0, x0p, dim=1) + EPS).pow(gamma).sum()
        elif dist_method == "cosine":
            # Internal term (Est vs Est)
            s2 = (cosine_dist(x, xp, dim=1) + EPS).pow(gamma).sum()
            # Internal term (True vs True)
            s3 = (cosine_dist(x0, x0p, dim=1) + EPS).pow(gamma).sum()
        else:
            raise ValueError(f"dist_method can be one of geodesic, chordal, and cosine.")
        loss = s1 - s2/2 - s3/2

    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss


def energy_loss_geodesic(x_true, x_est, verbose=True, kernel_func=powered_exponential(c=1, alpha=1), gamma=1):
    r"""
    Computes the Energy Score Loss for probabilistic predictions using geodesic distance with a specified kernel function for strict propriety.

    This function calculates a generalized, kernel-based Energy Score for angular data. 
    It computes the shortest-path (geodesic) distances between the estimated ensemble 
    and the ground truth, applies a specified kernel function (e.g., from `anglepy.kernels`), 
    and evaluates the proper scoring rule.

    The kernelized Energy Loss is formulated as:

    .. math::
        \mathbb{E}[k(d(\hat{x}, x)^\gamma)] - \frac{1}{2} \mathbb{E}[k(d(\hat{x}_i, \hat{x}_j)^\gamma)]

    where :math:`x` is the ground truth, :math:`\hat{x}` are the ensemble predictions, :math:`m` is the 
    number of ensemble members, :math:`d(\cdot, \cdot)` is the geodesic distance, and :math:`k(\cdot)` 
    is the specified `kernel_func`.

    Args:
        x_true (torch.Tensor): The ground truth angles in radians. Expected to be a tensor 
            that can be reshaped to `(Batch, Dim)`.
        x_est (torch.Tensor or list of torch.Tensor): The ensemble of predicted angles in radians. 
            If a tensor, it is split across the batch dimension. If a list, it is stacked 
            to form a shape of `(Batch, Samples, Dim)`.
        verbose (bool, optional): If True, returns a concatenated tensor containing the 
            total loss, the distance to ground truth (Term 1), and the pairwise ensemble 
            distance (Term 2). If False, returns only the total scalar loss. Defaults to True.
        kernel_func (callable, optional): A kernel function initialized with its hyperparameters 
            (e.g., `powered_exponential` from `anglepy.kernels`). Defaults to 
            `powered_exponential(c=1, alpha=1)`.
        gamma (float, optional): A power parameter applied to the geodesic distances 
            before they are passed into the kernel function. Defaults to 1.

    Returns:
        torch.Tensor: 
            - If `verbose=True`: A 1D tensor of shape `(3,)` containing 
              `[total_kernel_loss, term_1, term_2]`.
            - If `verbose=False`: A scalar tensor containing the computed total loss.

    Note:
        - This function expects inputs strictly in the range :math:`[0, 2\pi)`.
        - It relies on external functions (`vectorize`, `geodesic_dist`, `apply_kernel`) 
          and defaults to a kernel from `anglepy.kernels`, which must be available in the local scope.
    """
    # Tiny epsilon to prevent NaN gradients when d=0 and alpha < 1
    EPS = 1e-7 
    
    # Input Processing
    x_true = vectorize(x_true).unsqueeze(1) # Shape: [Batch, 1, Dim]
    
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))
    m = len(x_est)
    x_est = [vectorize(x_est[i]).unsqueeze(1) for i in range(m)]
    x_est = torch.cat(x_est, dim=1) # Shape: [Batch, Samples, Dim]
    
    # Term 1: Distance between Estimated Samples and Ground Truth
    d_xt = geodesic_dist(x_est, x_true) # Result: [B, m]
   
    # Apply kernel
    s1 = apply_kernel(d_xt.pow(gamma), kernel_func).mean()
    #s1 = -torch.exp(-((d_xt + EPS) / c).pow(alpha)).mean()

    # Term 2: Pairwise distances within Estimated Samples
    x_est_a = x_est.unsqueeze(2)
    x_est_b = x_est.unsqueeze(1)
    
    d_xx = geodesic_dist(x_est_a, x_est_b)
    
    # Apply kernel to internal distances with bias correction
    # s2 = -torch.exp(-((d_xx + EPS) / c).pow(alpha)).mean() * m / (m - 1)
    s2 = apply_kernel(d_xx.pow(gamma), kernel_func).mean() * m / (m - 1)

    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2 / 2)


def energy_loss_two_sample_geodesic(x0, x, xp, x0p=None, verbose=True, weights=None, kernel_func=powered_exponential(c=1, alpha=1), gamma=1):
    r"""
    Two-Sample Energy Loss using the specified kernel and geodesic distance.
    Assumes inputs are in radians :math:`[0, 2\pi)`.
    """
    EPS = 1e-7
    
    # Ensure inputs are tensors
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)

    if weights is None:
        weights = 1 / x0.size(0)

    # Helper function to cleanly apply the correct kernel
    # def apply_kernel(d):
    #     return -torch.exp(-((d + EPS) / c).pow(alpha))

    if x0p is None:
        # Cross terms (True vs Est)
        
        d_x_x0 = geodesic_dist(x, x0, dim=1)
        d_xp_x0 = geodesic_dist(xp, x0, dim=1)
      
        s1 = (apply_kernel(d_x_x0.pow(gamma), kernel_func) * weights).sum() / 2 + \
             (apply_kernel(d_xp_x0.pow(gamma), kernel_func) * weights).sum() / 2
        
        # Internal term (Est vs Est)
        d_x_xp = geodesic_dist(x, xp, dim=1)
        
        s2 = (apply_kernel(d_x_xp.pow(gamma), kernel_func) * weights).sum()
        loss = s1 - s2/2
        
    else:
        x0p = vectorize(x0p)
        
        # Cross terms (True vs Est)
        c1 = apply_kernel(geodesic_dist(x, x0, dim=1).pow(gamma), kernel_func).sum()
        c2 = apply_kernel(geodesic_dist(xp, x0, dim=1).pow(gamma), kernel_func).sum()
        c3 = apply_kernel(geodesic_dist(x, x0p, dim=1).pow(gamma), kernel_func).sum()
        c4 = apply_kernel(geodesic_dist(xp, x0p, dim=1).pow(gamma), kernel_func).sum()
        
        s1 = (c1 + c2 + c3 + c4) / 4
        
        
        # Internal term (Est vs Est)
        s2 = apply_kernel(geodesic_dist(x, xp, dim=1).pow(gamma), kernel_func).sum()
        # Internal term (True vs True)
        s3 = apply_kernel(geodesic_dist(x0, x0p, dim=1).pow(gamma), kernel_func).sum()
        
        loss = s1 - s2/2 - s3/2

    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss

