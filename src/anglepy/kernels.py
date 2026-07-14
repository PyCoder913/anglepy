'''
This module contains the kernel implementations to make the geodesic energy score a strictly proper scoring rule. For more details, refer to Table 1 of https://projecteuclid.org/journalArticle/Download?urlId=10.3150%2F12-BEJSP06.
'''

import torch

EPS = 1e-8

def apply_kernel(d, kernel_func):
    """
    Unified function.
    Returns the negative of the provided kernel function.
    """
    if kernel_func is None:
        return d
    return -kernel_func(d)

def powered_exponential(c=1, alpha=1):
    '''
    c > 0, 0 < alpha <= 1
    '''
    def kernel(d):
        return torch.exp(-((d + EPS) / c).pow(alpha))
    return kernel

def generalized_cauchy(c=1, tau=1, alpha=1):
    '''
    c > 0, tau > 0, 0 < alpha <= 1
    '''
    def kernel(d):
        return (1 + ((d + EPS) / c).pow(alpha)).pow(-tau / alpha)
    return kernel

def dagum(c=1, tau=1, alpha=0.5):
    '''
    c > 0, 0 < tau <= 1, 0 < alpha < tau 
    '''
    def kernel(d):
        x_tau = ((d + EPS) / c).pow(tau)
        return 1 - (x_tau / (1 + x_tau)).pow(alpha / tau)
    return kernel

def multiquadric(tau=1, delta=0.5):
    '''
    tau > 0, 0 < delta < 1
    '''
    def kernel(d):
        # Cosine is safe from zero-gradient issues, so EPS is unnecessary here
        return (1 - delta)**(2 * tau) / (1 + delta**2 - 2 * delta * torch.cos(d)).pow(tau)
    return kernel

def sine_power(alpha=1):
    '''
    0 < alpha < 2
    '''
    def kernel(d):
        # Added .abs() because fractional powers of negative numbers result in NaN
        return 1 - torch.sin(d / 2).abs().pow(alpha)
    return kernel

def spherical(c=1):
    '''
    c > 0
    '''
    def kernel(d):
        x = d / c
        positive_part = torch.clamp(1 - x, min=0.0) # (1 - d/c)+
        return (1 + 0.5 * x) * positive_part.pow(2)
    return kernel

def askey(c=1, tau=2):
    '''
    c > 0, tau >= 2
    '''
    def kernel(d):
        x = d / c
        positive_part = torch.clamp(1 - x, min=0.0)
        return positive_part.pow(tau)
    return kernel

def c2_wendland(c=torch.pi, tau=4):
    '''
    0 < c <= pi, tau >= 4
    '''
    def kernel(d):
        x = d / c
        positive_part = torch.clamp(1 - x, min=0.0)
        return (1 + tau * x) * positive_part.pow(tau)
    return kernel

def c4_wendland(c=torch.pi, tau=6):
    '''
    0 < c <= pi, tau >= 6
    '''
    def kernel(d):
        x = d / c
        positive_part = torch.clamp(1 - x, min=0.0)
        return (1 + tau * x + ((tau**2 - 1) / 3) * x.pow(2)) * positive_part.pow(tau)
    return kernel
