'''
This module implements the stochastic neural network blocks and circular embedding layers.
'''

import torch
import torch.nn as nn
from .data_loader import make_dataloader
from .utils import *

class CircularEmbeddingLayer(nn.Module):
    """
    Transforms specified angular or cyclic features into continuous representations using trigonometric embeddings.
    Used internally within ANGLE when the indices of circular covariates are specified.

    This layer separates the input tensor features into linear (non-circular) and circular components 
    based on the provided `circular_indices`. For each circular feature :math:`x_i`, it computes :math:`\cos(x_i)` 
    and :math:`\sin(x_i)` to project the feature onto a unit circle. This resolves discontinuities at period 
    boundaries (e.g., the jump from :math:`359^\circ` back to :math:`0^\circ`, or :math:`2\pi` to :math:`0`). The transformed 
    circular features are then concatenated with the unmodified linear features.

    Args:
        circular_indices (list of int or None): The column indices of the input tensor that correspond 
            to circular/cyclic features. Duplicate indices are ignored. If `None` or empty, the input 
            tensor is returned unmodified.

    Shape:
        - Input: :math:`(N, F)` where :math:`N` is the batch size and :math:`F` is the total number of input features.
        - Output: :math:`(N, F + C)` where :math:`C` is the number of unique, valid indices in `circular_indices`. 
          The output feature dimension is composed of :math:`(F - C)` linear features, followed by :math:`C` 
          cosine-transformed features, and finally :math:`C` sine-transformed features.

    Example:
        >>> # Suppose feature at index 1 is an angle in radians
        >>> layer = CircularEmbeddingLayer(circular_indices=[1])
        >>> x = torch.tensor([[1.5, 3.14159],   # batch 1: [linear, pi]
        ...                   [2.5, 0.0]])      # batch 2: [linear, 0]
        >>> layer(x)
        tensor([[ 1.5000, -1.0000,  0.0000],    # [linear, cos(pi), sin(pi)]
                [ 2.5000,  1.0000,  0.0000]])   # [linear, cos(0), sin(0)]
    """
    def __init__(self, circular_indices):
        super(CircularEmbeddingLayer, self).__init__()
        self.circular_indices = sorted(list(set(circular_indices))) if circular_indices is not None else []

    def forward(self, x):
        if not self.circular_indices:
            return x
            
        num_features = x.shape[1]
        linear_indices = [i for i in range(num_features) if i not in self.circular_indices]
        
        x_linear = x[:, linear_indices] if len(linear_indices) > 0 else torch.empty((x.shape[0], 0), device=x.device)
        x_circular = x[:, self.circular_indices]
        
        x_cos = torch.cos(x_circular)
        x_sin = torch.sin(x_circular)
        
        return torch.cat([x_linear, x_cos, x_sin], dim=1)


class StoLayer(nn.Module):    
    """
    A stochastic neural network layer that concatenates random noise to input features.

    This layer acts as a stochastic generator. It appends random noise (either Gaussian 
    or Uniform) to the input covariates and maps the concatenated tensor through a 
    linear layer. It can be used for conditional generation (given input features) 
    or unconditional generation (given only a batch size integer). If the input 
    covariates have fewer dimensions than expected, the remaining dimensions are 
    automatically padded with additional noise.

    Args:
        in_dim (int): Expected dimension of the input covariates. Set to 0 for 
            unconditional generation.
        out_dim (int): Dimension of the output tensor.
        noise_dim (int, optional): Dimension of the stochastic noise to inject. 
            Defaults to 100.
        add_bn (bool, optional): If True, appends a 1D Batch Normalization layer 
            after the linear transformation. Defaults to False.
        out_act (str or None, optional): String identifier for the activation function 
            to apply to the output (e.g., "relu", "sigmoid"). If set to "softmax" 
            and `out_dim` is 1, it automatically converts to "sigmoid". Defaults to None.
        noise_std (float, optional): Scaling factor (standard deviation for Gaussian, 
            or maximum amplitude for Uniform) applied to the generated noise. Defaults to 1.
        noise_dist (str, optional): The statistical distribution of the noise. 
            Must be either "gaussian" or "uniform". Defaults to "gaussian".
        verbose (bool, optional): If True, prints a warning when the input feature 
            dimension is less than `in_dim`. Defaults to True.

    Shape:
        - Input: A tensor of shape :math:`(N, F)` for conditional generation, where :math:`N` is 
          the batch size and :math:`F \le \text{in\_dim}`. Alternatively, an integer :math:`N` 
          representing the batch size for unconditional generation (requires :math:`\text{in\_dim} = 0`).
        - Output: A tensor of shape :math:`(N, \text{out\_dim})`.
    """
    def __init__(self, in_dim, out_dim, noise_dim=100, add_bn=False, out_act=None, noise_std=1, noise_dist="gaussian", verbose=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_std = noise_std
        self.verbose = verbose
        self.noise_dist = noise_dist
        
        layer = [nn.Linear(in_dim + noise_dim, out_dim)]
        if add_bn:
            layer += [nn.BatchNorm1d(out_dim)]
        self.layer = nn.Sequential(*layer)
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)
    
    def forward(self, x):
        device = next(self.layer.parameters()).device
        if isinstance(x, int):
            # For unconditional generation, x is the batch size.
            assert self.in_dim == 0
            if self.noise_dist == "gaussian":
                out = torch.randn(x, self.noise_dim, device=device) * self.noise_std
            elif self.noise_dist == "uniform":
                out = torch.rand(x, self.noise_dim, device=device) * self.noise_std
            else:
                raise ValueError(f"`noise_dist` can be either `gaussian` or `uniform`, `{self.noise_dist}` is not accepted.")
        else:
            if x.size(1) < self.in_dim and self.verbose:
                print("Warning: covariate dimension does not aligned with the specified input dimension; filling in the remaining dimension with noise.")
            if self.noise_dist == "gaussian":
                eps = torch.randn(x.size(0), self.noise_dim + self.in_dim - x.size(1), device=device) * self.noise_std
            elif self.noise_dist == "uniform":
                eps = torch.rand(x.size(0), self.noise_dim + self.in_dim - x.size(1), device=device) * self.noise_std
            else:
                raise ValueError(f"`noise_dist` can be either `gaussian` or `uniform`, `{self.noise_dist}` is not accepted.")
            out = torch.cat([x, eps], dim=1)
        out = self.layer(out)
        if self.out_act is not None:
            out = self.out_act(out)
        return out


def get_act_func(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid() 
    elif name == "tanh":
        return nn.Tanh() 
    elif name == "softmax":
        return nn.Softmax(dim=1)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "softplus":
        return nn.Softplus()
    else:
        return None


class StoResBlock(nn.Module):
    """A stochastic residual net block.

    Args:
        dim (int, optional): input dimension. Defaults to 100.
        hidden_dim (int, optional): hidden dimension (default to dim). Defaults to None.
        out_dim (int, optional): output dimension (default to dim). Defaults to None.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add batch normalization. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
    """
    def __init__(self, dim=100, hidden_dim=None, out_dim=None, noise_dim=100, add_bn=False, out_act=None, noise_std=1, noise_dist="gaussian"):
        super().__init__()
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.noise_dist = noise_dist
        if hidden_dim is None:
            hidden_dim = dim
        if out_dim is None:
            out_dim = dim
        self.layer1 = [nn.Linear(dim + noise_dim, hidden_dim)]
        self.add_bn = add_bn
        if add_bn:
            self.layer1.append(nn.BatchNorm1d(hidden_dim))
        self.layer1.append(nn.ReLU())
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Linear(hidden_dim + noise_dim, out_dim)
        if add_bn and out_act == "relu": # for intermediate blocks
            self.layer2 = nn.Sequential(*[self.layer2, nn.BatchNorm1d(out_dim)])
        if out_dim != dim:
            self.layer3 = nn.Linear(dim, out_dim)
        self.dim = dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)

    def forward(self, x):
        if self.noise_dim > 0:
            if self.noise_dist == "gaussian":
                eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
                out = self.layer1(torch.cat([x, eps], dim=1))
                eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
                out = self.layer2(torch.cat([out, eps], dim=1))
            elif self.noise_dist == "uniform":
                eps = torch.rand(x.size(0), self.noise_dim, device=x.device) * self.noise_std
                out = self.layer1(torch.cat([x, eps], dim=1))
                eps = torch.rand(x.size(0), self.noise_dim, device=x.device) * self.noise_std
                out = self.layer2(torch.cat([out, eps], dim=1))
            else:
                raise ValueError(f"`noise_dist` can be either `gaussian` or `uniform`, `{self.noise_dist}` is not accepted.")
        else:
            out = self.layer2(self.layer1(x))
        if self.out_dim != self.dim:
            out2 = self.layer3(x)
            out = out + out2
        else:
            out += x
        if self.out_act is not None:
            out = self.out_act(out)
        return out


class FiLMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, condition_dim, 
                 hidden_dim=512, noise_dim=0, add_bn=False, resblock=False, 
                 out_act=None, film_pos='out', film_level=1):
        super().__init__()
        self.film_pos = film_pos
        self.film_level = film_level
        film_out_dim = out_dim if film_pos == 'out' else in_dim
        if film_level > 1:
            self.condition_layer = nn.Linear(condition_dim, film_out_dim * 2)
        elif film_level == 1:
            self.condition_layer = nn.Linear(condition_dim, film_out_dim)
        if resblock:
            self.net = StoLayer(in_dim, out_dim, noise_dim, add_bn, out_act)
        else:
            self.net = StoResBlock(in_dim, hidden_dim, out_dim, noise_dim, add_bn, out_act)
        
    def forward(self, x, condition):
        out = self.net(x) if self.film_pos == 'out' else x
        if self.film_level > 1:
            gamma, beta = self.condition_layer(condition).chunk(2, dim=1)         
            out = gamma * out + beta
        elif self.film_level == 1:
            beta = self.condition_layer(condition)
            out = out + beta
        if self.film_pos == 'in':
            out = self.net(out)
        return out


class StoNetBase(nn.Module):
    def __init__(self, forward_sampling=True):
        super().__init__()
        self.sampling_func = self.forward if forward_sampling else self.sampling_func
    
    @torch.no_grad()
    def predict(self, x, target=["mean"], sample_size=100):
        """Point prediction.

        Args:
            x (torch.Tensor): input data
            target (str or float or list, optional): quantities to predict (mean/median/quantiles). float refers to the quantiles. Defaults to ["mean"].
            sample_size (int, optional): sample sizes for each x. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions
                - [:,:,i] gives the i-th sample of all x.
                - [i,:,:] gives all samples of x_i.
            
        Modified prediction for Circular Data.
        Calculates the circular mean of the generated samples.
        """
        # Get samples (Angles in [0, 2pi))
        # Shape: (data_size, out_dim, sample_size)
        samples = self.sample(x=x, sample_size=sample_size, expand_dim=True)
        
        if not isinstance(target, list):
            target = [target]
            
        results = []
        
        for t in target:
            if t == "mean":
                # Circular mean calculation
                # Convert angles back to unit vectors
                sin_vals = torch.sin(samples)
                cos_vals = torch.cos(samples)
                
                # Average the vectors (sample_size dimension)
                mean_sin = sin_vals.mean(dim=-1)
                mean_cos = cos_vals.mean(dim=-1)
                
                # Convert averaged vector back to angle [0, 2pi)
                circ_mean = torch.atan2(mean_sin, mean_cos) % (2 * torch.pi)
                results.append(circ_mean)
                
            else:
                if t == "median":
                    circ_median = get_circular_medians(samples)
                    results.append(circ_median)
                else:
                    assert isinstance(t, float)
                    point_quantiles = get_circular_quantiles(samples, t)
                    results.append(point_quantiles)

        if len(results) == 1:
            return results[0]
        else:
            return results

    
    def sample_onebatch(self, x, sample_size=100, expand_dim=True, require_grad=False):
        """Sampling new response data (for one batch of data).

        Args:
            x (torch.Tensor): new data of predictors of shape [data_size, covariate_dim]
            sample_size (int, optional): new sample size. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size) if expand_dim else (data_size*sample_size, response_dim), where response_dim could have multiple channels.
        """
        data_size = x.size(0) ## input data size
        if not require_grad:
            with torch.no_grad():
                # repeat the data for sample_size times, get a tensor [data, data, ..., data]
                x_rep = x.repeat(sample_size, 1)
                # samples of shape (data_size*sample_size, response_dim) such that samples[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size
                samples = self.sampling_func(x_rep).detach()

                if getattr(self, "unbounded", False):
                    samples = samples % (2 * torch.pi)
        else:
            x_rep = x.repeat(sample_size, 1)
            samples = self.sampling_func(x_rep)
            if getattr(self, "unbounded", False):
                samples = samples % (2 * torch.pi)
        if not expand_dim:# or sample_size == 1:
            return samples
        else:
            expand_dim = len(samples.shape)
            samples = samples.unsqueeze(expand_dim) # (data_size*sample_size, response_dim, 1)
            # a list of length data_size, each element is a tensor of shape (data_size, response_dim, 1)
            samples = list(torch.split(samples, data_size)) 
            samples = torch.cat(samples, dim=expand_dim) # (data_size, response_dim, sample_size)
            return samples
            # without expanding dimensions:
            # samples.reshape(-1, *samples.shape[1:-1])
    
    def sample_batch(self, x, sample_size=100, expand_dim=True, batch_size=None):
        """Sampling with mini-batches; only used when out-of-memory.

        Args:
            x (torch.Tensor): new data of predictors of shape [data_size, covariate_dim]
            sample_size (int, optional): new sample size. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.
            batch_size (int, optional): batch size. Defaults to None.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size) if expand_dim else (data_size*sample_size, response_dim), where response_dim could have multiple channels.
        """
        if batch_size is not None and batch_size < x.shape[0]:
            test_loader = make_dataloader(x, batch_size=batch_size, shuffle=False)
            samples = []
            for (x_batch,) in test_loader:
                samples.append(self.sample_onebatch(x_batch, sample_size, expand_dim))
            samples = torch.cat(samples, dim=0)
        else:
            samples = self.sample_onebatch(x, sample_size, expand_dim)
        return samples
    
    def sample(self, x, sample_size=100, expand_dim=True, verbose=True):
        """Sampling that adaptively adjusts the batch size according to the GPU memory."""
        batch_size = x.shape[0]
        while True:
            try:
                samples = self.sample_batch(x, sample_size, expand_dim, batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    if verbose:
                        print("Out of memory; reduce the batch size to {}".format(batch_size))
        return samples
    
    
class StoNet(StoNetBase):
    """
    A stochastic neural network.

    This network supports advanced architectural features including residual blocks, 
    Sufficient Dimension Reduction (SDR) bottlenecks, and native handling of circular 
    (angular) inputs and outputs. It maps input covariates to a specified output space 
    while injecting stochastic noise (Gaussian or Uniform) into the architecture.

    If `circular_indices` are provided, the input covariates are first passed through 
    a `CircularEmbeddingLayer` to resolve angular discontinuities. If `sdr=True`, the 
    embedded inputs are projected into a lower-dimensional subspace (`reduced_dim`) 
    via a bias-free linear bottleneck before entering the main stochastic layers.

    Args:
        in_dim (int): The number of raw input features (covariates).
        out_dim (int): The target number of output features. Note that the actual 
            network output dimension may change based on `circular_projection`.
        num_layer (int, optional): The total number of hidden layers. If `resblock=True`, 
            this must be an even number (automatically adjusted if odd). Defaults to 2.
        hidden_dim (int, optional): The number of neurons in each hidden layer. Defaults to 100.
        noise_dim (int, optional): The dimension of the stochastic noise vector. Defaults to 100.
        noise_std (float, optional): Scaling factor for the injected noise. Defaults to 1.
        add_bn (bool, optional): If True, applies Batch Normalization. Defaults to False.
        out_act (str or None, optional): Activation function for the final output layer 
            (e.g., "relu", "sigmoid"). Defaults to None.
        resblock (bool, optional): If True, groups hidden layers into residual blocks 
            (each containing 2 layers). Defaults to False.
        noise_all_layer (bool, optional): If True, injects noise into every hidden layer 
            rather than just the input. Defaults to True.
        out_bias (bool, optional): If True, includes a bias term in the final linear 
            output layer. Defaults to True.
        verbose (bool, optional): Enables warning logs (e.g., when correcting odd 
            `num_layer` for resblocks). Defaults to True.
        forward_sampling (bool, optional): Toggle for base class sampling behavior. Defaults to True.
        circular_indices (list of int or None, optional): Feature indices of the input 
            that should be treated as circular variables. Defaults to None.
        sdr (bool, optional): If True, applies a Sufficient Dimension Reduction (SDR) 
            linear bottleneck to the inputs before the hidden layers. Defaults to False.
        reduced_dim (int or None, optional): The bottleneck dimension for SDR. Must be 
            specified if `sdr=True` and must be less than or equal to the embedded 
            input dimension. Defaults to None.
        noise_dist (str, optional): Distribution of the noise ("gaussian" or "uniform"). 
            Defaults to "gaussian".
        unbounded (bool, optional): If True, bypasses circular projection rules for 
            the output layer. Defaults to False.
        circular_projection (str, optional): The method used to predict circular outputs. 
            Options are:
            - 'atan2': Doubles the output dimension. The network predicts (u, v) 
              coordinates which can be mapped to an angle using arctangent.
            - 'sigmoid': Outputs standard dimensions, assumed to be squashed to [0, 1].
            Defaults to 'atan2'.

    Attributes:
        network_out_dim (int): The actual output dimension of the final linear layer. 
            This equals `out_dim * 2` if `unbounded=False` and `circular_projection='atan2'`, 
            otherwise it equals `out_dim`.
        embedder (CircularEmbeddingLayer): The initial layer handling circular input features.
        beta_layer (nn.Linear): The SDR bottleneck layer, instantiated only if `sdr=True`.
        num_blocks (int or None): The number of residual blocks (i.e., `num_layer // 2`).
    """
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, 
                 noise_dim=100, noise_std=1, add_bn=False, out_act=None, resblock=False, 
                 noise_all_layer=True, out_bias=True, verbose=True, forward_sampling=True, 
                 circular_indices=None, sdr=False, reduced_dim=None, noise_dist="gaussian", unbounded=False, circular_projection='atan2'):
        super().__init__(forward_sampling=forward_sampling)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dist = noise_dist
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.add_bn = add_bn
        self.noise_all_layer = noise_all_layer
        self.out_bias = out_bias
        self.sdr = sdr 
        self.reduced_dim = reduced_dim
        self.unbounded = unbounded
        self.circular_projection = circular_projection
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)

        if self.unbounded:
            self.network_out_dim = out_dim
        else:
            if self.circular_projection == 'atan2':
                # Double the output dimension (u, v)
                # The network predicts 2 values for every 1 circular target
                self.network_out_dim = out_dim * 2
            elif self.circular_projection == 'sigmoid':
                self.network_out_dim = out_dim
            else:
                print(f'Only `atan2` or `sigmoid` is accepted for `circular_projection`, but got: `{self.circular_projection}`. Reverting to `atan2`...')
                self.circular_projection = 'atan2'
                self.network_out_dim = out_dim * 2
                

        self.circular_indices = circular_indices or []
        self.embedder = CircularEmbeddingLayer(self.circular_indices)
        # Calculate new input dimension after expanding circular vars to sin/cos
        embedded_in_dim = in_dim + len(self.circular_indices)

        # Conditionally build the SDR bottleneck matrix
        if self.sdr:
            if self.reduced_dim is None:
                raise ValueError("If sdr=True, reduced_dim must be specified as an integer.")
            if self.reduced_dim > embedded_in_dim:
                raise ValueError(f"reduced_dim ({self.reduced_dim}) cannot be greater than (embedded) input dimension ({embedded_in_dim}).")
            if self.reduced_dim <= 0:
                raise ValueError("reduced_dim must be strictly positive.")
                
            # Linear projection without bias to represent \beta^T X
            # Maps from embedded_in_dim to reduced_dim
            self.beta_layer = nn.Linear(embedded_in_dim, self.reduced_dim, bias=False)
            effective_in_dim = self.reduced_dim  # The stochastic layers now take the reduced dimension
        else:
            effective_in_dim = embedded_in_dim # Standard MLP behavior
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                print("The number of layers must be an even number for residual blocks. Changed to {}".format(str(num_layer)))
            num_blocks = num_layer // 2
            self.num_blocks = num_blocks
        self.resblock = resblock
        self.num_layer = num_layer
        
        if self.resblock: 
            if self.num_blocks == 1:
                self.net = StoResBlock(dim=effective_in_dim, hidden_dim=hidden_dim, out_dim=self.network_out_dim, 
                                       noise_dim=noise_dim, noise_dist=self.noise_dist, noise_std=noise_std, add_bn=add_bn, out_act=out_act)
            else:
                self.input_layer = StoResBlock(dim=effective_in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, 
                                               noise_dim=noise_dim, noise_dist=self.noise_dist, noise_std=noise_std, add_bn=add_bn, out_act="relu")
                if not noise_all_layer:
                    noise_dim = 0
                self.inter_layer = nn.Sequential(*[StoResBlock(dim=hidden_dim, noise_dim=noise_dim, noise_dist=self.noise_dist, noise_std=noise_std, add_bn=add_bn, out_act="relu")]*(self.num_blocks - 2))
                self.out_layer = StoResBlock(dim=hidden_dim, hidden_dim=hidden_dim, out_dim=self.network_out_dim, noise_dist=self.noise_dist,
                                             noise_dim=noise_dim, noise_std=noise_std, add_bn=add_bn, out_act=out_act) # output layer with concatinated noise
        else:
            self.input_layer = StoLayer(in_dim=effective_in_dim, out_dim=hidden_dim, noise_dim=noise_dim, noise_dist=self.noise_dist, noise_std=noise_std, add_bn=add_bn, out_act="relu", verbose=verbose)
            if not noise_all_layer:
                noise_dim = 0
            self.inter_layer = nn.Sequential(*[StoLayer(in_dim=hidden_dim, out_dim=hidden_dim, noise_dim=noise_dim, noise_dist=self.noise_dist, noise_std=noise_std, add_bn=add_bn, out_act="relu")]*(num_layer - 2))
            # self.out_layer = StoLayer(in_dim=hidden_dim, out_dim=out_dim, noise_dim=noise_dim, add_bn=False, out_act=out_act) # output layer with concatenated noise
            self.out_layer = nn.Linear(hidden_dim, self.network_out_dim, bias=out_bias)
            if self.out_act is not None:
                self.out_layer = nn.Sequential(*[self.out_layer, self.out_act])
            
    def forward(self, x):
        # Apply sin-cos embedding to circular covariates
        x = self.embedder(x)

        # Apply the bottleneck if SDR is enabled
        if self.sdr:
            x = self.beta_layer(x)
        
        if self.num_blocks == 1:
            out = self.net(x)
        else:
            out = self.out_layer(self.inter_layer(self.input_layer(x)))
            
        if self.unbounded:
            # Drop the final dimension if predicting a single scalar to match expected atan2 tensor shapes
            return out.squeeze(-1) if out.shape[-1] == 1 else out
        else:
            if self.circular_projection == 'atan2':
                # Original Cartesian Projection
                u = out[..., 0]
                v = out[..., 1]
                angles = torch.atan2(v, u) % (2 * torch.pi)
                return angles

            elif self.circular_projection == 'sigmoid':
                # Scaled Sigmoid Projection
                logits = out.squeeze(-1) if out.shape[-1] == 1 else out[..., 0] # Extract a single scalar logit per prediction
                angles = torch.sigmoid(logits) * 2 * torch.pi # Apply sigmoid to bound to (0, 1), then scale by 2*pi
                return angles


class Net(nn.Module):
    """Deterministic neural network.

    Args:
        in_dim (int, optional): input dimension. Defaults to 1.
        out_dim (int, optional): output dimension. Defaults to 1.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to False.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
    """
    def __init__(self, in_dim=1, out_dim=1, num_layer=2, hidden_dim=100, 
                 add_bn=False, sigmoid=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.add_bn = add_bn
        self.sigmoid = sigmoid
        
        net = [nn.Linear(in_dim, hidden_dim)]
        if add_bn:
            net += [nn.BatchNorm1d(hidden_dim)]
        net += [nn.ReLU(inplace=True)]
        for _ in range(num_layer - 2):
            net += [nn.Linear(hidden_dim, hidden_dim)]
            if add_bn:
                net += [nn.BatchNorm1d(hidden_dim)]
            net += [nn.ReLU(inplace=True)]
        net.append(nn.Linear(hidden_dim, out_dim))
        if sigmoid:
            out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
            net.append(out_act)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResMLPBlock(nn.Module):
    """MLP residual net block.

    Args:
        dim (int): dimension of input and output.
    """
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        out += x
        return self.relu(out)


class ResMLP(nn.Module):
    """Residual MLP.

    Args:
        in_dim (int, optional): input dimension. Defaults to 1.
        out_dim (int, optional): output dimension. Defaults to 1.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
    """
    def __init__(self, in_dim=1, out_dim=1, num_layer=2, hidden_dim=100, add_bn=False, sigmoid=False):
        super().__init__()
        out_act = "sigmoid" if sigmoid else None
        if num_layer % 2 != 0:
            num_layer += 1
            print("The number of layers must be an even number for residual blocks. Added one layer.")
        num_blocks = num_layer // 2
        self.num_blocks = num_blocks
        if num_blocks == 1:
            self.net = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                   noise_dim=0, add_bn=add_bn, out_act=out_act)
        else:
            self.input_layer = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, 
                                           noise_dim=0, add_bn=add_bn, out_act="relu")
            self.inter_layer = nn.Sequential(*[StoResBlock(dim=hidden_dim, noise_dim=0, add_bn=add_bn, out_act="relu")]*(self.num_blocks - 2))
            self.out_layer = StoResBlock(dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                         noise_dim=0, add_bn=add_bn, out_act=out_act)

    def forward(self, x):
        if self.num_blocks == 1:
            return self.net(x)
        else:
            return self.out_layer(self.inter_layer(self.input_layer(x)))
