import numpy as np
import torch
import torch.nn as nn
import math

__all__ = ["generate_atan2_circular_data", "generate_pn_circular_data", "preanm_simulator"]

def to_pi_range(theta):
    """Converts angles from [0, 2pi) to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi

def to_2pi_range(theta):
    """Converts angles from [-pi, pi] to [0, 2pi)."""
    return theta % (2 * np.pi)

def generate_atan2_circular_data(
    n_samples, 
    beta1,
    beta2,
    n_circular_cov=0,          
    noise_type='normal', 
    noise_scale=0.5, 
    random_seed=None,
    pi_range=False,
    a=4  # Added multiplier for the non-linear strength
):
    r"""
    Generates circular data using a deterministic Arctan2 mapping of 
    non-linearly interacting signals.

    This function constructs circular responses by simulating raw linear and 
    circular covariates, applying pre-additive noise, and embedding the circular 
    features via sine and cosine transformations. Base linear signals 
    (:math:`z_1, z_2`) are computed and then heavily distorted using a cross-term 
    :math:`\tanh` function controlled by parameter `a`:

    .. math::
        \mu_1 = z_1 + a \tanh(z_2)

    .. math::
        \mu_2 = z_2 + a \tanh(z_1)

    The final angular response is computed using the four-quadrant inverse 
    tangent mapping, :math:`\theta = \arctan2(\mu_2, \mu_1)`, projecting the 
    distorted Cartesian coordinates onto the unit circle.

    Parameters
    ----------
    n_samples : int
        The number of data points to generate.
    beta1 : array_like
        The coefficient weights for the first base linear signal (:math:`z_1`), 
        which acts as the denominator/x-component prior to mapping.
    beta2 : array_like
        The coefficient weights for the second base linear signal (:math:`z_2`), 
        which acts as the numerator/y-component prior to mapping. 
        Must be the exact same length as `beta1`.
    n_circular_cov : int, optional
        The number of circular covariates to generate (sampled from a von Mises 
        distribution). Remaining covariate slots defined by the length of the beta 
        arrays will be filled with standard normal linear covariates. Default is 0.
    noise_type : {'normal', 'uniform'}, optional
        The statistical distribution of the pre-additive noise applied to the 
        raw covariates before embedding. Default is 'normal'.
    noise_scale : float, optional
        The magnitude of the applied noise. If `noise_type` is 'normal', this acts 
        as the standard deviation. If 'uniform', this acts as the symmetric boundary 
        :math:`[-\text{scale}, \text{scale}]`. Default is 0.5.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility. Default is None.
    pi_range : bool, optional
        Determines the range of the output angles. If True, returns angles in the 
        range :math:`[-\pi, \pi]`. If False, returns angles in :math:`[0, 2\pi)`. Default is False.
    a : float, optional
        Multiplier controlling the strength of the non-linear cross-term interaction 
        between the base signals. Setting `a=0` results in a model driven purely 
        by linear combinations of the covariates (approach 1). Default is 4.

    Returns
    -------
    theta : ndarray
        1D array of shape `(n_samples,)` containing the generated circular response 
        variables (angles).
    X_original : ndarray
        2D array of shape `(n_samples, n_linear_cov + n_circular_cov)` containing 
        the raw, un-noised covariates. Linear covariates are positioned in the 
        first columns, followed by the raw circular covariates.

    Raises
    ------
    ValueError
        If `beta1` and `beta2` are not of the same length.
    ValueError
        If the beta arrays are too short to support the requested number of 
        circular covariates (each circular covariate requires 2 beta weights 
        for its sine/cosine embedding).
    ValueError
        If an unsupported `noise_type` is provided.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    b1 = np.array(beta1)
    b2 = np.array(beta2)
    
    if len(b1) != len(b2):
        raise ValueError("beta1 and beta2 must have the same length.")
        
    # Calculate expected dimensions
    total_beta_weights = len(b1)
    
    # Each circular covariate becomes 2 features (sin and cos)
    n_linear_cov = total_beta_weights - (2 * n_circular_cov)
    
    if n_linear_cov < 0:
        raise ValueError("beta arrays are too short to support the requested number of embedded circular covariates.")
        
    # Generate raw covariates
    if n_linear_cov > 0:
        X_lin_raw = np.random.standard_normal((n_samples, n_linear_cov))
    else:
        X_lin_raw = np.empty((n_samples, 0))
        
    if n_circular_cov > 0:
        # Generate raw circular covariates using von Mises
        X_circ_raw = np.random.vonmises(mu=0.0, kappa=1.0, size=(n_samples, n_circular_cov))
        X_circ_raw = X_circ_raw % (2 * np.pi)
    else:
        X_circ_raw = np.empty((n_samples, 0))
        
    # Un-embedded, raw inputs for the model to train on
    X_original = np.hstack([X_lin_raw, X_circ_raw])
    
    # Generate pre-additive noise
    if noise_type == 'normal':
        noise_lin = np.random.normal(0.0, noise_scale, size=(n_samples, n_linear_cov))
        noise_circ = np.random.normal(0.0, noise_scale, size=(n_samples, n_circular_cov))
    elif noise_type == 'uniform':
        noise_lin = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, n_linear_cov))
        noise_circ = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, n_circular_cov))
    else:
        raise ValueError("noise_type must be 'normal' or 'uniform'")
        
    # Apply noise to raw covariates
    X_eff_lin = X_lin_raw + noise_lin if n_linear_cov > 0 else X_lin_raw
    
    # Modulo addition ensures circular noise stays on the circle
    X_eff_circ = (X_circ_raw + noise_circ) % (2 * np.pi) if n_circular_cov > 0 else X_circ_raw
    
    # Apply sine/cosine embedding to the noisy circular covariates
    if n_circular_cov > 0:
        X_eff_circ_cos = np.cos(X_eff_circ)
        X_eff_circ_sin = np.sin(X_eff_circ)
        X_eff_embedded = np.hstack([X_eff_lin, X_eff_circ_cos, X_eff_circ_sin])
    else:
        X_eff_embedded = X_eff_lin
        
    # Base linear combinations
    z1 = X_eff_embedded @ b1  # Acts as the base 'x' denominator
    z2 = X_eff_embedded @ b2  # Acts as the base 'y' numerator
    
    # Apply stable nonlinear transformations using 'a'
    # If a=0, the model is linear. As 'a' increases, cross-terms heavily distort the signals.
    mu1 = z1 + a * np.tanh(z2)
    mu2 = z2 + a * np.tanh(z1)
    
    # Map to (-pi, pi] using Arctan2 with the newly distorted signals
    # np.arctan2(y, x) handles the correct quadrant mapping automatically
    theta_raw = np.arctan2(mu2, mu1)
    if not pi_range:
        # Shift the range from [-pi, pi] to [0, 2pi)
        theta = np.mod(theta_raw, 2 * np.pi)
    else:
        theta = theta_raw
        
    return theta, X_original
  

def generate_pn_circular_data(
    n_samples, 
    beta1, 
    beta2, 
    n_circular_cov=0,          
    noise_type='uniform', 
    noise_scale=0.05, 
    random_seed=None,
    pi_range=False,
    a=2  # Multiplier for the non-linear strength
):
    r"""
    Generates circular data from a Projected Normal (PN) model using Approach 3
    and 4, featuring pre-additive noise and non-linear covariate interactions.

    This function simulates circular data by first generating latent bivariate 
    normal vectors :math:`(Y_1, Y_2)` and projecting them onto the unit circle using 
    the four-quadrant inverse tangent, :math:`\theta = \arctan2(Y_2, Y_1)`. 
    
    The latent means :math:`(\mu_1, \mu_2)` are calculated by applying pre-additive 
    noise to the raw covariates, embedding any circular covariates into :math:`\cos` 
    and :math:`\sin` components, computing base linear projections :math:`(z_1, z_2)`, and 
    finally applying a stable non-linear cross-term transformation controlled 
    by the parameter `a`.

    Parameters
    ----------
    n_samples : int
        The number of data points to generate.
    beta1 : array_like
        The coefficient weights for the first latent dimension projection (:math:`z_1`).
    beta2 : array_like
        The coefficient weights for the second latent dimension projection (:math:`z_2`).
        Must be the exact same length as `beta1`.
    n_circular_cov : int, optional
        The number of circular covariates to generate (sampled from a von Mises 
        distribution). Remaining covariate slots defined by the length of the beta 
        arrays will be filled with standard normal linear covariates. Default is 0.
    noise_type : {'uniform', 'normal'}, optional
        The statistical distribution of the pre-additive noise applied to the 
        raw covariates. Default is 'uniform'.
    noise_scale : float, optional
        The magnitude of the applied noise. If `noise_type` is 'normal', this acts 
        as the standard deviation. If 'uniform', this acts as the symmetric boundary 
        :math:`[-\text{scale}, \text{scale}]`. Default is 0.05.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility. Default is None.
    pi_range : bool, optional
        Determines the range of the output angles. If True, returns angles in the 
        range :math:`[-\pi, \pi]`. If False, returns angles in :math:`[0, 2\pi)`. Default is False.
    a : float, optional
        Multiplier controlling the strength of the non-linear cross-term interaction 
        between the latent variables (e.g., :math:`\mu_1 = z_1 + a \tanh(z_2)`). 
        Setting `a=0` results in a strictly linear model (approach 3). Default is 2.

    Returns
    -------
    theta : ndarray
        1D array of shape `(n_samples,)` containing the generated circular response 
        variables (angles).
    X_original : ndarray
        2D array of shape `(n_samples, n_linear_cov + n_circular_cov)` containing 
        the un-noised, raw covariates. Linear covariates are positioned in the 
        first columns, followed by the raw circular covariates.

    Raises
    ------
    ValueError
        If `beta1` and `beta2` are not of the same length.
    ValueError
        If the beta arrays are too short to support the requested number of 
        circular covariates (each circular covariate requires 2 beta weights 
        for its sine/cosine embedding).
    ValueError
        If an unsupported `noise_type` is provided.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    b1 = np.array(beta1)
    b2 = np.array(beta2)
    
    if len(b1) != len(b2):
        raise ValueError("beta1 and beta2 must have the same length.")
        
    total_beta_weights = len(b1)
    n_linear_cov = total_beta_weights - (2 * n_circular_cov)
    
    if n_linear_cov < 0:
        raise ValueError("beta arrays are too short to support the requested covariates.")
        
    # Generate raw covariates
    if n_linear_cov > 0:
        X_lin_raw = np.random.standard_normal((n_samples, n_linear_cov))
    else:
        X_lin_raw = np.empty((n_samples, 0))
        
    if n_circular_cov > 0:
        X_circ_raw = np.random.vonmises(mu=0.0, kappa=1.0, size=(n_samples, n_circular_cov))
        X_circ_raw = X_circ_raw % (2 * np.pi)
        if pi_range:
            X_circ_raw = to_pi_range(X_circ_raw)
    else:
        X_circ_raw = np.empty((n_samples, 0))
        
    X_original = np.hstack([X_lin_raw, X_circ_raw])
    
    # Generate pre-additive noise
    if noise_type == 'normal':
        noise_lin = np.random.normal(0.0, noise_scale, size=(n_samples, n_linear_cov))
        noise_circ = np.random.normal(0.0, noise_scale, size=(n_samples, n_circular_cov))
    elif noise_type == 'uniform':
        noise_lin = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, n_linear_cov))
        noise_circ = np.random.uniform(-noise_scale, noise_scale, size=(n_samples, n_circular_cov))
    else:
        raise ValueError("noise_type must be 'normal' or 'uniform'")
        
    # Apply noise
    X_eff_lin = X_lin_raw + noise_lin if n_linear_cov > 0 else X_lin_raw
    X_eff_circ = (X_circ_raw + noise_circ) % (2 * np.pi) if n_circular_cov > 0 else X_circ_raw
    
    # Apply sine/cosine embedding to circular covariates
    if n_circular_cov > 0:
        X_eff_circ_cos = np.cos(X_eff_circ)
        X_eff_circ_sin = np.sin(X_eff_circ)
        X_eff_embedded = np.hstack([X_eff_lin, X_eff_circ_cos, X_eff_circ_sin])
    else:
        X_eff_embedded = X_eff_lin
        
    # Calculate base linear projections
    z1 = X_eff_embedded @ b1
    z2 = X_eff_embedded @ b2
    
    # Apply stable nonlinear transformations using 'a'
    # If a=0, the model is linear.
    # As 'a' increases, the cross-term interaction heavily distorts the linear space.
    mu1 = z1 + a * np.tanh(z2) 
    mu2 = z2 + a * np.tanh(z1) 
    
    # Generate latent bivariate Normal vectors (Y) and project
    Y1 = np.random.normal(loc=mu1, scale=1.0)
    Y2 = np.random.normal(loc=mu2, scale=1.0)
    theta_raw = np.arctan2(Y2, Y1)
    
    if not pi_range:
        theta = theta_raw % (2 * np.pi)
    else:
        theta = theta_raw
        
    return theta, X_original


def preanm_simulator(
    true_function="softplus", 
    n_train=5000, 
    n_eval=1000,      # Number of points for the evaluation grid
    x_lower=0, 
    x_upper=2, 
    x_eval_lower=-2,  
    x_eval_upper=6,   
    noise_std=1.0, 
    noise_dist="gaussian", 
    device=torch.device("cpu")
):
    r"""
    Simulates circular data from a Pre-Additive Noise Model (Pre-ANM) and computes 
    the theoretical true conditional circular mean for an evaluation grid.

    This simulator generates a training dataset where a base latent function :math:`f` 
    is applied to a noisy input :math:`X_n = X + \epsilon`, and the output is wrapped 
    onto the unit circle:
    
    .. math::
        Y = \arctan2(\sin(f(X + \epsilon)), \cos(f(X + \epsilon))) \pmod{2\pi}

    Alongside the scatter training data, it calculates the true conditional 
    circular mean :math:`\mathbb{E}[Y|X=x]` over a linearly spaced extrapolation grid. 
    Because the noise is pre-additive, the expected value requires marginalizing 
    over the noise distribution. This is achieved via a high-sample Monte Carlo 
    approximation at each grid point.
    
    Code is inspired by Shen & Meinshausen (2025).

    Parameters
    ----------
    true_function : str or callable, optional
        The underlying base function :math:`f(x)` to apply to the noisy covariates. 
        If a string, accepted values are:
        * `'softplus'`: Applies `nn.Softplus()`.
        * `'cubic'`: Applies :math:`x^3 / 3`.
        * `'square'`: Applies :math:`\text{ReLU}(x)^2 / 2`.
        * `'log'`: Applies a piecewise function: a linear combination for :math:`x \le 2` and :math:`\ln(1+x)` for :math:`x > 2`.
        If a callable, it should accept and return a PyTorch tensor. 
        Default is 'softplus'.
    n_train : int, optional
        The number of training samples to generate. The base covariates :math:`X` are 
        sampled uniformly. Default is 5000.
    n_eval : int, optional
        The number of linearly spaced points to generate for the evaluation grid. 
        Default is 1000.
    x_lower : float, optional
        The lower bound of the uniform sampling distribution for the training 
        covariates :math:`X`. Default is 0.
    x_upper : float, optional
        The upper bound of the uniform sampling distribution for the training 
        covariates :math:`X`. Default is 2.
    x_eval_lower : float, optional
        The lower bound of the evaluation/extrapolation grid. Default is -2.
    x_eval_upper : float, optional
        The upper bound of the evaluation/extrapolation grid. Default is 6.
    noise_std : float, optional
        The standard deviation :math:`\sigma` of the pre-additive noise :math:`\epsilon`. 
        Default is 1.0.
    noise_dist : {'gaussian', 'uniform'}, optional
        The statistical distribution of the pre-additive noise. If 'uniform', 
        it is scaled internally to match the standard deviation specified by 
        `noise_std`. Default is 'gaussian'.
    device : str or torch.device, optional
        The PyTorch device (e.g., 'cpu', 'cuda') on which the returned tensors 
        should be allocated. Default is 'cpu'.

    Returns
    -------
    x_train : torch.Tensor
        A 2D tensor of shape `(n_train, 1)` containing the clean, uniformly 
        sampled training covariates :math:`X`.
    y_train : torch.Tensor
        A 2D tensor of shape `(n_train, 1)` containing the generated circular 
        response variables :math:`Y` in the range :math:`[0, 2\pi)`.
    x_eval : torch.Tensor
        A 2D tensor of shape `(n_eval, 1)` containing the linearly spaced 
        evaluation/extrapolation grid.
    y_eval_mean_circ : torch.Tensor
        A 2D tensor of shape `(n_eval, 1)` containing the Monte Carlo 
        approximation of the true conditional circular mean at each point in 
        `x_eval`. Values are in the range :math:`[0, 2\pi)`.
    """
    if isinstance(true_function, str):
        if true_function == "softplus":
            base_function = lambda x: nn.Softplus()(x)
        elif true_function == "cubic":
            base_function = lambda x: x.pow(3)/3
        elif true_function == "square":
            base_function = lambda x: (nn.functional.relu(x)).pow(2)/2
        elif true_function == "log":
            base_function = lambda x: (x/3 + np.log(3) - 2/3)*(x <= 2) + (torch.log(1 + x*(x > 2)))*(x > 2) 
    else:
        base_function = true_function
            
    if isinstance(device, str):
        device = torch.device(device)

    # Generate training data (scatter)
    x_train = torch.rand(n_train, 1) * (x_upper - x_lower) + x_lower
    
    if noise_dist == "gaussian":
        eps_train = torch.randn(n_train, 1) * noise_std
    elif noise_dist == "uniform":
        eps_train = (torch.rand(n_train, 1) - 0.5) * noise_std * math.sqrt(12)
        
    xn_train = x_train + eps_train
    
    cosine_train = torch.cos(base_function(xn_train))
    sine_train = torch.sin(base_function(xn_train))
    y_train = torch.atan2(sine_train, cosine_train) % (2 * math.pi)

    # Generate true evaluation mean
    # Create a perfectly spaced grid for the line plot
    x_eval = torch.linspace(x_eval_lower, x_eval_upper, n_eval).unsqueeze(1)
    
    # Monte Carlo sampling to find the true conditional circular mean E[Y|X]
    gen_sample_size = 10000 
    x_rep = torch.repeat_interleave(x_eval, gen_sample_size, dim=0)
    
    if noise_dist == "gaussian":
        eps_rep = torch.randn(x_rep.size(0), 1) * noise_std
    elif noise_dist == "uniform":
        eps_rep = (torch.rand(x_rep.size(0), 1) - 0.5) * noise_std * math.sqrt(12)
        
    x_rep_noisy = x_rep + eps_rep
    
    # Evaluate true function on the noisy grid
    sines = torch.sin(base_function(x_rep_noisy))
    cosines = torch.cos(base_function(x_rep_noisy))
    y_eval_samples = torch.atan2(sines, cosines) % (2 * math.pi)
    
    # Collapse back using circular mean calculation
    y_eval_matrix = y_eval_samples.view(n_eval, gen_sample_size)
    mean_sin = torch.sin(y_eval_matrix).mean(dim=1, keepdim=True)
    mean_cos = torch.cos(y_eval_matrix).mean(dim=1, keepdim=True)
    
    y_eval_mean_circ = torch.atan2(mean_sin, mean_cos) % (2 * math.pi)
        
    return x_train.to(device), y_train.to(device), x_eval.to(device), y_eval_mean_circ.to(device)
