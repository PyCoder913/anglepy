import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from astropy.stats import kuiper_two
from scipy.stats import f as f_dist 



__all__ = ["circular_conditional_equality_test", "get_circular_medians", "get_circular_quantiles", "plot_donut", "plot_spokeplot"]

def circular_conditional_equality_test(X_1, y_1, X_2, y_2, angle_params, random_state=42):
    r"""
    Tests the equality of conditional distributions for circular responses using the ANGLE framework.

    This function evaluates the null hypothesis :math:`H_0: P_{Y|X}^{(1)} = P_{Y|X}^{(2)}` against 
    the alternative :math:`H_1: P_{Y|X}^{(1)} \neq P_{Y|X}^{(2)}`. It trains the ANGLE generative 
    model once on the first dataset and evaluates the generative mismatch on a randomly 
    partitioned second dataset using two-sample circular tests.

    Parameters
    ----------
    X_1 : array-like or torch.Tensor
        Covariates for the first sample, representing :math:`\mathcal{S}_1`.
    y_1 : array-like or torch.Tensor
        Angular responses for the first sample :math:`\mathcal{S}_1`, bounded in :math:`[0, 2\pi)`.
    X_2 : array-like or torch.Tensor
        Covariates for the second sample, representing :math:`\mathcal{S}_2`.
    y_2 : array-like or torch.Tensor
        Angular responses for the second sample :math:`\mathcal{S}_2`, bounded in :math:`[0, 2\pi)`.
    angle_params : dict
        Dictionary of hyperparameters used to initialize the ANGLE model.
    random_state : int, default=42
        Random seed for reproducibility in model initialization and dataset partitioning.

    Returns
    -------
    p_val_kuiper : float
        The :math:`p`-value obtained from the two-sample Kuiper's test.
    p_val_manova : float
        The :math:`p`-value obtained from the Trigonometric MANOVA two-sample test.

    Notes
    -----
    The testing procedure implements Algorithm 1 from the ANGLE framework:
    
    1. Fit the ANGLE network on :math:`\mathcal{S}_1` to obtain the generative model :math:`\tilde{g}_1`.
    2. Randomly partition :math:`\mathcal{S}_2` into two equal halves, :math:`\mathcal{S}_{21}` and :math:`\mathcal{S}_{22}`.
    3. Generate synthetic circular responses :math:`\tilde{Y}_{21,i} = \tilde{g}_1(X_{21,i}, \epsilon_i)` 
       for the covariates in :math:`\mathcal{S}_{21}`.
    4. Compare the synthetic responses :math:`\tilde{\mathcal{S}}_{21}` with the true held-out 
       responses :math:`\mathcal{S}_{22}` using standard circular two-sample tests. Under :math:`H_0`, 
       these sets of responses are approximately identically distributed.
    """
    from .ANGLE import ANGLE
    # Fit structured ANGLE on S1
    angle_S1 = ANGLE(
        X_1, y_1, 
        **angle_params,
        sdr=False,             
        random_state=random_state
    )
    
    # Randomly partition S2 into two equal halves
    X_21, X_22, y_21, y_22 = train_test_split(
        X_2, y_2, 
        test_size=0.5, 
        random_state=random_state
    )
    
    # Generate synthetic circular responses for S21
    y_21_synth_tensor = angle_S1.sample(X_21, sample_size=1)
    
    y_21_synth = y_21_synth_tensor.detach().cpu().numpy().flatten() % (2 * np.pi)
    y_22_true = y_22.detach().cpu().numpy().flatten() % (2 * np.pi)
    
    # Perform both tests: Kuiper and MANOVA
    _, p_val_kuiper = kuiper_two(y_21_synth, y_22_true)
    _, p_val_manova = trigonometric_manova_two_sample(y_21_synth, y_22_true)
    
    return p_val_kuiper, p_val_manova

def get_circular_medians(ensemble):
    r"""
    Calculates the circular median along the last dimension of the ensemble.
    
    Args:
        ensemble (torch.Tensor): A tensor of shape :math:`(..., \text{sample\_size})`.
            Predictions for a circular target in the range :math:`[0, 2\pi)`.
            
    Returns:
        torch.Tensor: A tensor of shape :math:`(...)` containing the computed point medians.
    """
    pi = torch.pi
    
    # Define candidates: (..., 2 * sample_size)
    antipodes = (ensemble + pi) % (2 * pi)
    candidates = torch.cat([ensemble, antipodes], dim=-1)
    
    # Expand dimensions to compute pairwise geodesic distances
    ensemble_exp = ensemble.unsqueeze(-1) # shape:   (..., sample_size, 1)
    candidates_exp = candidates.unsqueeze(-2) # shape: (..., 1, 2 * sample_size)
    
    # Calculate geodesic distance
    diff = torch.abs(ensemble_exp - candidates_exp) % (2 * pi)
    arc = torch.minimum(diff, 2 * pi - diff) 
    
    # Sum distances across the ensemble's sample dimension (dim=-2)
    total_distances = arc.sum(dim=-2) # shape: (..., 2 * sample_size)
    
    # Find the index of the best candidate
    best_idx = torch.argmin(total_distances, dim=-1, keepdim=True)
    
    # Gather the final medians and remove the trailing dummy dimension
    medians = torch.gather(candidates, dim=-1, index=best_idx).squeeze(-1) # shape: (...)
    return medians


def get_circular_quantiles(ensemble, a):
    r"""
    Calculates the circular conditional quantile along the last dimension.
    
    Args:
        ensemble (torch.Tensor): A tensor of shape :math:`(..., \text{sample\_size})`.
            Predictions for a circular target in the range :math:`[0, 2\pi)`.
        a (float): The quantile level in :math:`(0, 1)`.
                  
    Returns:
        torch.Tensor: A tensor of shape :math:`(...)` containing the computed point quantiles.
    """
    pi = torch.pi
    
    # Define candidates: (..., 2 * sample_size)
    antipodes = (ensemble + pi) % (2 * pi)
    candidates = torch.cat([ensemble, antipodes], dim=-1) 
    
    ensemble_exp = ensemble.unsqueeze(-1)
    candidates_exp = candidates.unsqueeze(-2)
    
    u = (ensemble_exp - candidates_exp) % (2 * pi)
    
    cost = torch.where(
        u < pi, 
        a * u, 
        (1 - a) * (2 * pi - u)
    )
    total_cost = cost.sum(dim=-2)
    best_idx = torch.argmin(total_cost, dim=-1, keepdim=True)
    quantiles = torch.gather(candidates, dim=-1, index=best_idx).squeeze(-1)
    return quantiles

def plot_donut(y_test, y_pred, ax, title="Donut Plot"):
    """
    Plots a donut plot on a given matplotlib Axes object.
    Considers North as 0 radians and angle increases clockwise.
    Reference: Jha, J., & Biswas, A. (2017). Multiple circular–circular regression. Statistical Modelling, 17(3), 142-171.
    """
    # Calculate the radius based on the cosine of the error
    r = 1 + torch.cos(y_pred - y_test)
    
    # Calculate Cartesian coordinates
    x = r * torch.cos(y_pred)
    y = r * torch.sin(y_pred)
    
    # Determine Deviation Direction
    diff = (y_pred - y_test + torch.pi) % (2 * torch.pi) - torch.pi
    cw_mask = diff < 0
    acw_mask = diff >= 0
    
    # Convert tensors to numpy for plotting
    x_np, y_np = x.numpy(), y.numpy()
    cw_mask_np, acw_mask_np = cw_mask.numpy(), acw_mask.numpy()
    
    # Draw the reference circles
    inner_circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=1, alpha=0.7)
    outer_circle = plt.Circle((0, 0), 2, color='black', fill=False, linewidth=1, alpha=0.7)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)
    
    # Scatter the points
    base_color = '#0072B2' 
    
    # Clockwise deviations: Solid circles
    ax.scatter(x_np[cw_mask_np], y_np[cw_mask_np], 
               facecolors=base_color, edgecolors='black', 
               marker='o', s=60, zorder=3, label='Clockwise Deviation')
    
    # Anticlockwise deviations: Empty circles
    ax.scatter(x_np[acw_mask_np], y_np[acw_mask_np], 
               facecolors='none', edgecolors=base_color, 
               linewidths=1.5, marker='o', s=60, zorder=3, label='Anticlockwise Deviation')
    
    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.axis('off')
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.set_title(title, y=-0.05, fontsize=14, fontweight='bold')


def plot_spokeplot(y_test, y_pred, ax, r_inner=1, r_outer=2, title="Spokeplot"):
    """
    Plots a spokeplot on a given matplotlib Axes object.
    Considers North as 0 radians and angle increases clockwise.
    """
    # Map to Cartesian coordinates (North = 0, Clockwise)
    x_test = r_outer * torch.sin(y_test)
    y_test_cart = r_outer * torch.cos(y_test)
    x_pred = r_inner * torch.sin(y_pred)
    y_pred_cart = r_inner * torch.cos(y_pred)
    
    x_test_np, y_test_cart_np = x_test.numpy(), y_test_cart.numpy()
    x_pred_np, y_pred_cart_np = x_pred.numpy(), y_pred_cart.numpy()
    
    # Draw the reference circles
    inner_circle = plt.Circle((0, 0), r_inner, color='black', fill=False, linewidth=1, alpha=0.7, zorder=1)
    outer_circle = plt.Circle((0, 0), r_outer, color='black', fill=False, linewidth=1, alpha=0.7, zorder=1)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)
    ax.scatter([0], [0], color='#333333', s=30, zorder=2) # Center dot
    
    # Draw the connecting lines (spokes)
    for i in range(len(y_test)):
        ax.plot([x_test_np[i], x_pred_np[i]], 
                [y_test_cart_np[i], y_pred_cart_np[i]], 
                color='gray', alpha=0.5, linewidth=1.2, zorder=2)

    # Scatter the points
    color_true = '#0072B2'
    color_pred = '#D55E00'
    
    ax.scatter(x_test_np, y_test_cart_np, 
               facecolors=color_true, edgecolors='black', 
               s=60, zorder=3, label='True (Observed)')
    ax.scatter(x_pred_np, y_pred_cart_np, 
               facecolors=color_pred, edgecolors='black', 
               s=60, zorder=3, label='Predicted')

    # Formatting
    ax.set_aspect('equal')
    padding = 0.4
    ax.set_xlim(-r_outer - padding, r_outer + padding)
    ax.set_ylim(-r_outer - padding, r_outer + padding)
    ax.axis('off') 
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1), frameon=False, fontsize=10)
    ax.text(0, r_outer + 0.1, 'N (0 rad)', ha='center', va='bottom', fontsize=11)
    ax.set_title(title, y=-0.05, fontsize=14, fontweight='bold')

def trigonometric_manova_two_sample(x1, x2):
    r"""
    Computes the trigonometric MANOVA (Hotelling's :math:`T^2`) test for two circular samples.

    This function tests the null hypothesis :math:`H_0` that the mean vectors of two 
    independent circular samples are equal against the alternative :math:`H_1` that they 
    are different. It operates by embedding the angular data into a bivariate Cartesian 
    space using trigonometric transformations and then applying the standard two-sample 
    Hotelling's :math:`T^2` test.

    Parameters
    ----------
    x1 : array-like
        First sample of circular observations, represented as angles in radians.
    x2 : array-like
        Second sample of circular observations, represented as angles in radians.

    Returns
    -------
    f_stat : float
        The computed :math:`F`-statistic derived from Hotelling's :math:`T^2` value.
    p_value : float
        The corresponding :math:`p`-value evaluated from the :math:`F`-distribution.

    Notes
    -----
    The procedure involves the following mathematical steps:

    1. Transform the angular observations into bivariate coordinates: 
       :math:`Y = [\cos(x), \sin(x)]`.
    2. Compute the sample mean vectors and the sample covariance matrices for both groups.
    3. Calculate the pooled covariance matrix :math:`S_{pool}`.
    4. Compute Hotelling's :math:`T^2` statistic:
       
       .. math::
          T^2 = \\frac{n_1 n_2}{n_1 + n_2} (\bar{Y}_1 - \bar{Y}_2)^T S_{pool}^{-1} (\bar{Y}_1 - \bar{Y}_2)

    5. Convert the :math:`T^2` statistic to an :math:`F`-statistic, which follows an 
       :math:`F`-distribution with degrees of freedom :math:`df_1 = p` and :math:`df_2 = n - p - 1`, 
       where :math:`p = 2` and :math:`n = n_1 + n_2`.
    """
    n1 = len(x1)
    n2 = len(x2)
    n = n1 + n2
    p = 2  # Two dimensions: cosine and sine

    # Transform angles to bivariate Cartesian coordinates
    Y1 = np.column_stack([np.cos(x1), np.sin(x1)])
    Y2 = np.column_stack([np.cos(x2), np.sin(x2)])

    # Calculate group means
    mean1 = np.mean(Y1, axis=0)
    mean2 = np.mean(Y2, axis=0)

    # Calculate sample covariance matrices (ddof=1 for unbiased estimator)
    S1 = np.cov(Y1, rowvar=False, ddof=1)
    S2 = np.cov(Y2, rowvar=False, ddof=1)

    # Compute the pooled covariance matrix
    S_pool = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n - 2)

    # Compute Hotelling's T^2 statistic
    mean_diff = mean1 - mean2
    inv_S_pool = np.linalg.pinv(S_pool)
    t2 = (n1 * n2 / n) * (mean_diff @ inv_S_pool @ mean_diff.T)

    # 6. Convert T^2 to the F-statistic
    f_stat = t2 * (n - p - 1) / ((n - 2) * p)

    # Calculate P-value using the F-distribution 
    df1 = p
    df2 = n - p - 1
    p_value = f_dist.sf(f_stat, df1, df2) 

    return f_stat, p_value

def vectorize(x, multichannel=False):
    """Vectorize data in any shape.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)
        
def cor(x, y):
    """Compute the correlation between two signals.

    Args:
        x (torch.Tensor): input data
        y (torch.Tensor): input data

    Returns:
        torch.Tensor: correlation between x and y
    """
    x = vectorize(x)
    y = vectorize(y)
    x = x - x.mean(0)
    y = y - y.mean(0)
    return ((x * y).mean()) / (x.std(unbiased=False) * y.std(unbiased=False))

def make_folder(name):
    """Make a folder.

    Args:
        name (str): folder name.
    """
    if not os.path.exists(name):
        print('Creating folder: {}'.format(name))
        os.makedirs(name)

def check_for_gpu(device):
    """Check if a CUDA device is available.

    Args:
        device (torch.device): current set device.
    """
    if device.type == "cuda":
        if torch.cuda.is_available():
            print("GPU is available, running on GPU.\n")
        else:
            print("GPU is NOT available, running instead on CPU.\n")
    else:
        if torch.cuda.is_available():
            print("Warning: You have a CUDA device, so you may consider using GPU for potential acceleration\n by setting device to 'cuda'.\n")
        else:
            print("Running on CPU.\n")
