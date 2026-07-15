Welcome to anglepy's documentation!
===================================

**anglepy** is a specialized Python package designed for deep distributional regression with directional and circular data. It provides the deep engression-based architecture, kernels, metrics, and customized loss functions needed to effectively build, train, and evaluate circular data neural networks.

``anglepy`` offers a unified framework to handle the unique geometric properties of angular data within modern machine learning workflows. It offers the following:

* Prediction with model-intrinsic uncertainty quantification
* Extrapolation on the circle
* Sufficient dimension reduction
* Testing equality of conditional distributions

Getting Started
===============

To install the latest stable release, simply run the following command in your terminal:

.. code-block:: bash

   pip install anglepy

Explore the Documentation
=========================

.. grid:: 2
   :gutter: 3
   :margin: 0

   .. grid-item-card:: 🛠️ Installation
      :link: install
      :link-type: doc

      Step-by-step instructions to set up ``anglepy`` and its dependencies in your environment.

   .. grid-item-card:: 📚 API Reference
      :link: api
      :link-type: doc

      Detailed documentation of the architectures, kernels, loss functions, and metrics.

   .. grid-item-card:: 💻 Usage Example
      :link: examples/Example_Notebook
      :link-type: doc

      A Jupyter notebook demonstrating core workflows and package functionality.

   .. grid-item-card:: 📝 Cite
      :link: cite
      :link-type: doc

      Information on how to cite the paper in your work.

.. toctree::
   :hidden:

   install
   api
   examples/Example_Notebook
   cite


Authors
=======

.. grid:: 3
   :gutter: 3
   :margin: 0

   .. grid-item-card:: 👤 Rajdeep Pathak
      :link: https://sites.google.com/view/rajdeeppathak/
      :link-type: url
      :text-align: center

   .. grid-item-card:: 👤 Archi Roy
      :link: https://sites.google.com/view/archi-roy
      :link-type: url
      :text-align: center

   .. grid-item-card:: 👤 Tanujit Chakraborty
      :link: https://www.ctanujit.org/
      :link-type: url
      :text-align: center


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`