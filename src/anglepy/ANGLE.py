'''
This module implements the main ANGLE API, which includes the `CircularEngressor` class and the ANGLE function for fitting 
the model to data. It allows for multivariate predictors and circular response variables, with options for standardization, 
noise injection, and various distance metrics in the loss function.
'''

import torch
import numpy as np
from .losses import *
from .models import StoNet
from .kernels import *
from .data_loader import make_dataloader
from .utils import *
import random
from .utils import cor

__all__ = ["ANGLE", "CircularEngressor"]

def set_random_seed(seed):
    """Sets the random seed for deterministic behavior in PyTorch and NumPy."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Enforce deterministic behavior in cuDNN (for GPU runs)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ANGLE(x, y, classification=False,
               num_layer=2, hidden_dim=100, noise_dim=100, noise_std=1, noise_dist="gaussian", out_act=None,
               add_bn=True, resblock=False, gamma=1, kernel_func=None,
               lr=0.0001, num_epochs=500, batch_size=None, 
               print_every_nepoch=100, print_times_per_epoch=1,
               device="cpu", standardize=True, verbose=True, dist_method="geodesic",
                  noise_all_layer=True, circular_indices=None, sdr=False, reduced_dim=None, random_state=None, unbounded=False, circular_projection='atan2'): 
    """API to fit ANGLE to the data. It allows multivariate predictors and circular response variables. 
    Variables are per default internally standardized (training with standardized data, while predictions and evaluations are on original scale).

    Args:
        x (torch.Tensor): training data of predictors.
        y (torch.Tensor): training data of circular responses in [0, 2\pi).
        classification (bool, optional): classification or not. Defaults to False.
        num_layer (int, optional): number of (linear) layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        noise_std (float, optional): standard deviation of the injected noise. Defaults to 1.
        noise_dist (str, optional): distribution of the noise. Choices = ["gaussian", "uniform"]. Defaults to "gaussian".
        out_act (str, optional): output activation function. Defaults to None.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        resblock (bool, optional): whether to use residual blocks (skip connections). Defaults to False.
        gamma (float, optional): power parameter in the energy loss. Defaults to 1.
        kernel_func (callable, optional): Instantiated negative of strictly positive definite kernel 
            function applied to the distance metric. Defaults to None.
            See kernels.py for available kernels, e.g., powered_exponential(c=1, alpha=1).
        lr (float, optional): learning rate. Defaults to 0.005.
        num_epochs (int, optional): number of epochs. Defaults to 500.
        batch_size (int, optional): batch size. Defaults to None, referring to the full batch.
        print_every_nepoch (int, optional): print losses every print_every_nepoch number of epochs. Defaults to 100.
        print_times_per_epoch (int, optional): print losses for print_times_per_epoch times per epoch. Defaults to 1.
        device (str, torch.device, optional): device. Defaults to "cpu". Choices = ["cpu", "gpu", "cuda"].
        standardize (bool, optional):  whether to standardize data during training. Defaults to True.
        verbose (bool, optional): whether to print losses and info. Defaults to True.
        dist_method (str, optional): underlying distance metric in the GCES loss. Choices = ["chordal", "geodesic", "cosine"]. Defaults to "chordal". 
        noise_all_layer (bool, optional): whether to inject noise into every hidden layer (pre-ANM) or only the last layer (post-ANM). Defaults to True (pre-ANM).
        circular_indices (list of int, optional): indices specifying which input features are circular. Defaults to None.
        sdr (bool, optional): whether to use Single-Index / Multiple-Index mode for sufficient dimension reduction. Defaults to False.
        reduced_dim (int, optional): target dimension size when using SDR mode. Defaults to None.
        random_state (int, optional): random seed for reproducibility. Defaults to None.
        unbounded (bool, optional): whether to output in modulated (unbounded) mode. Defaults to False.
        circular_projection (str, optional): method used to project outputs to the circle when bounded. Choices = ['atan2', 'sigmoid']. Defaults to 'atan2'.

    Returns:
        CircularEngressor: a fitted circular engression model.
    """
    if x.shape[0] != y.shape[0]:
        raise Exception("The sample sizes for the covariates and response do not match. Please check.")
    angle = CircularEngressor(in_dim=x.shape[1], out_dim=y.shape[1], classification=classification, 
                          num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim, noise_std=noise_std, noise_dist=noise_dist,
                          out_act=out_act, resblock=resblock, add_bn=add_bn, gamma=gamma, kernel_func=kernel_func,
                          lr=lr, num_epochs=num_epochs, batch_size=batch_size, 
                          standardize=standardize, device=device, check_device=verbose, verbose=verbose, 
                          dist_method=dist_method, circular_indices=circular_indices, sdr=sdr, reduced_dim=reduced_dim,
                          random_state=random_state, unbounded=unbounded, noise_all_layer=noise_all_layer, circular_projection=circular_projection)
    angle.train(x, y, num_epochs=num_epochs, batch_size=batch_size, 
                    print_every_nepoch=print_every_nepoch, print_times_per_epoch=print_times_per_epoch, 
                    standardize=standardize, verbose=verbose)
    return angle


class CircularEngressor(object):
    """Circular Engressor class.

    A distributional regression model tailored for circular responses. Minimizes 
    the generalized circular energy score loss.

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        classification (bool, optional): classification or not. Defaults to False.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        noise_std (float, optional): standard deviation of the injected noise. Defaults to 1.
        noise_dist (str, optional): distribution of the noise. Choices = ["gaussian", "uniform"]. Defaults to "gaussian".
        out_act (str or bool, optional): output activation function. Defaults to False.
        resblock (bool, optional): whether to use residual blocks (skip-connections). Defaults to False.
        add_bn (bool, optional): whether to add Batch Normalization layers. Defaults to True.
        gamma (float, optional): power parameter in the energy loss. Defaults to 1.
        kernel_func (callable, optional): Instantiated negative of strictly positive definite kernel 
            function applied to the distance metric. Defaults to powered_exponential(c=1, alpha=1).
            See kernels.py for available kernels.
        lr (float, optional): learning rate. Defaults to 0.005.
        num_epochs (int, optional): number of epochs for training. Defaults to 500.
        batch_size (int, optional): batch size. Defaults to None, referring to the full batch.
        standardize (bool, optional): whether to standardize data during training. Defaults to True.
        device (str or torch.device, optional): compute device. Choices = ["cpu", "gpu", "cuda"]. Defaults to "cpu".
        check_device (bool, optional): whether to verify the selected device is available. Defaults to True.
        verbose (bool, optional): whether to print training progress and logs. Defaults to True.
        dist_method (str, optional): underlying distance metric in the GCES loss. Choices = ["chordal", "geodesic", "cosine"]. Defaults to "chordal". 
        circular_indices (list of int, optional): indices specifying which input features are circular. Defaults to None.
        sdr (bool, optional): whether to use Single-Index / Multiple-Index mode for sufficient dimension reduction. Defaults to False.
        reduced_dim (int, optional): target dimension size when using SDR mode. Defaults to None.
        random_state (int, optional): random seed for reproducibility. Defaults to None.
        unbounded (bool, optional): whether to output in modulated (unbounded) mode. Defaults to False.
        noise_all_layer (bool, optional): whether to inject noise into every hidden layer (pre-ANM) or only the last layer (post-ANM). Defaults to True (pre-ANM).
        circular_projection (str, optional): method used to project outputs to the circle when bounded. 
            Choices = ['atan2', 'sigmoid']. Defaults to 'atan2'.
    """
    def __init__(self, 
                 in_dim, out_dim, classification=False,
                 num_layer=2, hidden_dim=100, noise_dim=100, noise_std=1, noise_dist="gaussian",
                 out_act=False, resblock=False, add_bn=True, gamma=1, kernel_func=powered_exponential(c=1, alpha=1),
                 lr=0.0001, num_epochs=500, batch_size=None, standardize=True, 
                 device="cpu", check_device=True, verbose=True, dist_method="geodesic", 
                 circular_indices=None, sdr=False, reduced_dim=None, random_state=None, unbounded=False, noise_all_layer=True, circular_projection='atan2'): 
        super().__init__()
        self.classification = classification
        if classification:
            out_act = "softmax"
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.noise_dist = noise_dist
        self.out_act = out_act
        self.resblock = resblock
        self.add_bn = add_bn
        self.gamma = gamma
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dist_method = dist_method
        self.random_state = random_state
        self.unbounded = unbounded
        self.noise_all_layer = noise_all_layer
        self.circular_projection = circular_projection
        self.kernel_func = kernel_func
                     
        if isinstance(device, str):
            if device == "gpu" or device == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device(device)
        self.device = device
        self.sdr = sdr 
        self.reduced_dim = reduced_dim 

        if self.random_state is not None:
          set_random_seed(self.random_state)
          
        self.standardize = standardize
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        
        self.model = StoNet(in_dim, out_dim, num_layer, hidden_dim, noise_dim, noise_std, add_bn, out_act, resblock, noise_dist=self.noise_dist, circular_indices=circular_indices,
                           sdr=self.sdr, reduced_dim=self.reduced_dim, unbounded=self.unbounded, noise_all_layer=self.noise_all_layer, circular_projection=self.circular_projection).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.verbose = verbose
        
        self.tr_loss = None
                   
    @property
    def beta_proj(self):
        """
        Returns the estimated gamma matrix if the model has SDR enabled.
        The returned tensor has shape (reduced_dim, embedded_in_dim).
        """
        if not self.sdr:
            raise ValueError("beta_proj is only available when sdr=True.")
        
        # PyTorch nn.Linear stores weights internally as (out_features, in_features).
        # Therefore, the shape is natively (reduced_dim, embedded_in_dim).
        return self.model.beta_layer.weight.detach()
      
    def train_mode(self):
        self.model.train()
        
    def eval_mode(self):
        self.model.eval()
        
    def summary(self):
        """Print the model architecture and hyperparameters."""
        print("Engression model with\n" +
              "\t number of layers: {}\n".format(self.num_layer) +
              "\t hidden dimensions: {}\n".format(self.hidden_dim) +
              "\t noise dimensions: {}\n".format(self.noise_dim) +
              "\t residual blocks: {}\n".format(self.resblock) +
              "\t number of epochs: {}\n".format(self.num_epochs) +
              "\t batch size: {}\n".format(self.batch_size) +
              "\t learning rate: {}\n".format(self.lr) +
              "\t standardization: {}\n".format(self.standardize) +
              "\t training mode: {}\n".format(self.model.training) +
              "\t device: {}\n".format(self.device))
        print("Training loss (original scale):\n" +
              "\t energy-loss: {:.2f}, \n\tE(|Y-Yhat|): {:.2f}, \n\tE(|Yhat-Yhat'|): {:.2f}".format(
                  self.tr_loss[0], self.tr_loss[1], self.tr_loss[2]))
        
    def _standardize_data_and_record_stats(self, x, y):
        """Standardize the data and record the mean and standard deviation of the training data.
        Avoid standardizing the circular target y.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor): training data of responses.

        Returns:
            torch.Tensor: standardized data.
        """
        self.x_mean = torch.mean(x, dim=0)
        self.x_std = torch.std(x, dim=0)
        self.x_std[self.x_std == 0] += 1e-5
        x_standardized = (x - self.x_mean) / self.x_std
        y_standardized = y # Avoid standardization of circular target
        self.x_mean = self.x_mean.to(self.device)
        self.x_std = self.x_std.to(self.device)
        return x_standardized, y_standardized

    def standardize_data(self, x, y=None):
        """Standardize the data, if self.standardize is True.
        Avoid standardizing the circular target y.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor, optional): training data of responses. Defaults to None.

        Returns:
            torch.Tensor: standardized or original data.
        """
        if y is None:
            if self.standardize:
                return (x - self.x_mean) / self.x_std
            else:
                return x
        else:
            if self.standardize:
                return (x - self.x_mean) / self.x_std, y
            else:
                return x, y
    
    def unstandardize_data(self, y, x=None, expand_dim=False):
        """Transform the predictions back to the original scale.
        
        For circular data:
        - y (targets): Remains unchanged (Standardization was skipped).
        - x (predictors): Unstandardized using recorded mean/std.
        """
        # Handle Predictors (x) - Unstandardize if they exist and standardization is on
        if x is not None and self.standardize:
            x = x * self.x_std + self.x_mean

        # Handle Targets (y) - Do nothing
        # We just return y as is.
        
        if x is None:
            return y
        else:
            return x, y
        
    def train(self, x, y, num_epochs=None, batch_size=None, lr=None, print_every_nepoch=100, print_times_per_epoch=1, standardize=None, verbose=True):
        """Fit the model.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor): trainging data of responses.
            num_epochs (int, optional): number of training epochs. Defaults to None.
            batch_size (int, optional): batch size for mini-batch SGD. Defaults to None.
            lr (float, optional): learning rate.
            print_every_nepoch (int, optional): print losses every print_every_nepoch number of epochs. Defaults to 100.
            print_times_per_epoch (int, optional): print losses for print_times_per_epoch times per epoch. Defaults to 1.
            standardize (bool, optional): whether to standardize the data. Defaults to True.
            verbose (bool, optional): whether to print losses and info. Defaults to True.
        """
        self.train_mode()
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is None:
            batch_size = self.batch_size if self.batch_size is not None else x.size(0)
        if lr is not None:
            if lr != self.lr:
                self.lr = lr
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if standardize is not None:
            self.standardize = standardize
            
        x = vectorize(x)
        y = vectorize(y)
        if self.standardize:
            if verbose:
                print("Data is standardized for training only; the printed training losses are on the standardized scale. \n" +
                    "However during evaluation, the predictions, evaluation metrics, and plots will be on the original scale.\n" +
                     "Note that the circular target variable y is not standardized.\n")
            x, y = self._standardize_data_and_record_stats(x, y)
        x = x.to(self.device)
        y = y.to(self.device)
        
        if batch_size >= x.size(0)//2:
            if verbose:
                print("Batch is larger than half of the sample size. Training based on full-batch gradient descent.")
            self.batch_size = x.size(0)
            for epoch_idx in range(self.num_epochs):
                self.model.zero_grad()
                y_sample1 = self.model(x)
                y_sample2 = self.model(x)
                if self.dist_method == "geodesic":
                    loss, loss1, loss2 = energy_loss_two_sample_geodesic(y, y_sample1, y_sample2, verbose=True, kernel_func=self.kernel_func, gamma=self.gamma)
                # elif self.dist_method == "geodesic" and kernel_func is None:
                #     loss, loss1, loss2 = energy_loss_two_sample(y, y_sample1, y_sample2, gamma=self.gamma, 
                #                                                 verbose=True, dist_method=self.dist_method)
                else:
                    loss, loss1, loss2 = energy_loss_two_sample(y, y_sample1, y_sample2, gamma=self.gamma, 
                                                                verbose=True, dist_method=self.dist_method)
                loss.backward()
                self.optimizer.step()
                if (epoch_idx == 0 or  (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                    print("[Epoch {} ({:.0f}%)] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                        epoch_idx + 1, 100 * epoch_idx / self.num_epochs, loss.item(), loss1.item(), loss2.item()))
        else:
            train_loader = make_dataloader(x, y, batch_size=batch_size, shuffle=True)
            if verbose:
                print("Training based on mini-batch gradient descent with a batch size of {}.".format(batch_size))
            for epoch_idx in range(self.num_epochs):
                self.zero_loss()
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    self.train_one_iter(x_batch, y_batch)
                    if (epoch_idx == 0 or (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                        if (batch_idx + 1) % ((len(train_loader) - 1) // print_times_per_epoch) == 0:
                            self.print_(epoch_idx, batch_idx)

        # Evaluate performance on the training data (on the original scale)
        self.model.eval()
        x, y = self.unstandardize_data(y, x)
        self.tr_loss = self.eval_loss(x, y, loss_type="energy", verbose=True)
        
        if verbose:
            print("\nTraining loss on the original (non-standardized) scale:\n" +
                "\tEnergy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                    self.tr_loss[0], self.tr_loss[1], self.tr_loss[2]))
            
        if verbose:
            print("\nPrediction-loss E(|Y-Yhat|) and variance-loss E(|Yhat-Yhat'|) should ideally be equally large" +
                "\n-- consider training for more epochs or adjusting hyperparameters if there is a mismatch ")
    
    def zero_loss(self):
        self.tr_loss = 0
        self.tr_loss1 = 0
        self.tr_loss2 = 0
    
    def train_one_iter(self, x_batch, y_batch):
        self.model.zero_grad()
        y_sample1 = self.model(x_batch)
        y_sample2 = self.model(x_batch)

        if self.dist_method == "geodesic":
            loss, loss1, loss2 = energy_loss_two_sample_geodesic(y_batch, y_sample1, y_sample2, verbose=True, kernel_func=self.kernel_func, gamma=self.gamma)
        # elif self.dist_method == "geodesic" and kernel_func is None:
        #     loss, loss1, loss2 = energy_loss_two_sample(y_batch, y_sample1, y_sample2, verbose=True, dist_method=self.dist_method, gamma=self.gamma)
        else:
            loss, loss1, loss2 = energy_loss_two_sample(y_batch, y_sample1, y_sample2, verbose=True, dist_method=self.dist_method, gamma=self.gamma)
        loss.backward()
        self.optimizer.step()
        self.tr_loss += loss.item()
        self.tr_loss1 += loss1.item()
        self.tr_loss2 += loss2.item()
        
    def print_loss(self, epoch_idx, batch_idx, return_loss=False):
        loss_str = "[Epoch {} ({:.0f}%), batch {}] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
            epoch_idx + 1, 100 * epoch_idx / self.num_epochs, batch_idx + 1, 
            self.tr_loss / (batch_idx + 1), self.tr_loss1 / (batch_idx + 1), self.tr_loss2 / (batch_idx + 1))
        if return_loss:
            return loss_str
        else:
            print(loss_str)
    
    @torch.no_grad()
    def predict(self, x, target="mean", sample_size=100):
        """Point prediction. 

        Args:
            x (torch.Tensor): data of predictors.
            target (str or float or list, optional): a quantity of interest to predict. float refers to the quantiles. Defaults to "mean".
            sample_size (int, optional): generated sample sizes for each x. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions.
        """
        self.eval_mode()  
        x = vectorize(x)
        x = x.to(self.device)
        x = self.standardize_data(x)
        y_pred = self.model.predict(x, target, sample_size)
        #if isinstance(y_pred, list):
            #for i in range(len(y_pred)):
                #y_pred[i] = self.unstandardize_data(y_pred[i])
        #else:
            #y_pred = self.unstandardize_data(y_pred)
        return y_pred
    
    @torch.no_grad()
    def sample(self, x, sample_size=100, expand_dim=True):
        """Sample new response data.

        Args:
            x (torch.Tensor): test data of predictors.
            sample_size (int, optional): generated sample sizes for each x. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size).
                - [:,:,i] consists of the i-th sample of all x.
                - [i,:,:] consists of all samples of x_i.
        """
        self.eval_mode()
        x = vectorize(x)
        x = x.to(self.device)
        x = self.standardize_data(x)
        y_samples = self.model.sample(x, sample_size, expand_dim=expand_dim)            
        y_samples = self.unstandardize_data(y_samples, expand_dim=expand_dim)
        if sample_size == 1:
            y_samples = y_samples.squeeze(len(y_samples.shape) - 1)
        return y_samples
    
    @torch.no_grad()
    def eval_loss(self, x, y, loss_type="l2", sample_size=None, gamma=1, verbose=False):
        """Compute the loss for evaluation.

        Args:
            x (torch.Tensor): data of predictors.
            y (torch.Tensor): data of responses.
            loss_type (str, optional): loss type. Defaults to "l2". Choices: ["l2", "l1", "energy", "cor"].
            sample_size (int, optional): generated sample sizes for each x. Defaults to 100.
            gamma (float, optional): gamma in energy score. Defaults to 1.
        
        Returns:
            float: evaluation loss.
        """
        if sample_size is None:
            sample_size = 2 if loss_type == "energy" else 100
        self.eval_mode()
        x = vectorize(x)
        y = vectorize(y)
        x = x.to(self.device)
        y = y.to(self.device)
        if loss_type == "l2":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = (y - y_pred).pow(2).mean()
        elif loss_type == "cor":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = cor(y, y_pred)
        elif loss_type == "l1":
            y_pred = self.predict(x, target=0.5, sample_size=sample_size)
            loss = (y - y_pred).abs().mean()
        else:
            assert loss_type == "energy"
            y_samples = self.sample(x, sample_size=sample_size, expand_dim=False)
            if self.dist_method == "geodesic":
                loss = energy_loss_geodesic(y, y_samples, verbose=verbose, kernel_func=self.kernel_func, gamma=gamma)
            # elif self.dist_method == "geodesic" and self.kernel_func is None:
            #     loss = energy_loss(y, y_samples, gamma=gamma, 
            #                    verbose=verbose, dist_method=self.dist_method)
            else:
                loss = energy_loss(y, y_samples, gamma=gamma, 
                               verbose=verbose, dist_method=self.dist_method)
        if not verbose:
            return loss.item()
        else:
            loss, loss1, loss2 = loss
            return loss.item(), loss1.item(), loss2.item()        
