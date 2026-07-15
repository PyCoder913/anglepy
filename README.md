# ANGLE: Angular Neural Generative Learning via Engression

**Authors:** Rajdeep Pathak, Archi Roy, Tanujit Chakraborty

[![Paper](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/abs/2607.12833) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

`anglepy` is a lightweight deep generative Python framework designed for non-parametric distributional regression on circular data. Traditional regression targets the conditional mean, which can be geometrically misleading for circular responses under multimodal, skewed, or asymmetric data structures. ANGLE addresses these limitations by learning the full conditional distribution of an angular response, given Euclidean and circular covariates, through a generative map optimized via a generalized circular energy score (GCES) loss.



## Key Features

* **Intrinsic Uncertainty Quantification:** Provides prediction with model-intrinsic uncertainty quantification, bounding true poses with predictive intervals.
* **Extrapolation on the Circle:** Extends its utility to underexplored challenges like out-of-distribution extrapolation.
* **Sufficient Dimension Reduction (SDR):** Finds low-dimensional representations of high-dimensional covariates without discarding predictive information.
* **Conditional Distribution Equality Testing:** Provides unified methodologies to test equality across conditional distributions.
* **Flexible Architecture:** Seamlessly accommodates both pre-additive and post-additive noise models (covariate noise and response noise).
* **Computationally Efficient:** Maintains a bandwidth-free architecture that is significantly more lightweight than existing Bayesian alternatives.

## ⚙️ Installation

You can install the package directly via PyPI or clone the repository to install it from the source.

### Option 1: Install via PyPI (Coming soon)
```bash
pip install anglepy
```

### Option 2: Install from Source
If you want to modify the code or run the latest development version, you can clone the repository:
```bash
git clone https://github.com/PyCoder913/anglepy.git
cd anglepy
pip install -r requirements.txt
```

### Quick Start
For detailed interactive examples, please check the Jupyter notebook in the `examples/` directory.

## 🚀 Applications

The practical efficacy of the ANGLE framework has been rigorously demonstrated across diverse data modalities:

* **📸 Object Pose Estimation:** Evaluated on the PASCAL3D+ benchmark to predict the horizontal rotation angle (azimuth) of 12 object categories from imagery, utilizing fine-tuned visual encoders like Inception-v3 and ConvNeXt.
* **🌬️ Wind Direction Prediction:** Applied to complex meteorological datasets from Germany and India to estimate the full distribution of wind directions based on spatial coordinates, providing critical probabilistic predictions for safety-critical operations.

## Documentation
Full documentation, including API references, mathematical foundations, and detailed usage tutorials, can be found in the docs/ folder or hosted online (link coming soon).

## 📝 Citation
If you use this code, models, or find our work helpful in your research, please consider citing our paper:
```bibtex
@article{pathak2026angle,
  title={ANGLE: Angular Neural Generative Learning via Engression},
  author={Pathak, Rajdeep and Roy, Archi and Chakraborty, Tanujit},
  journal={arXiv preprint arXiv:2607.12833},
  year={2026}
}

