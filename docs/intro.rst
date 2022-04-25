Neural Lifetimes
^^^^^^^^^^^^^^^^

Introduction
------------
One of the most important problems a firm faces is the question of consumer
value. Over the years, there have been many attempts to address this issue, with
one of the most successful being the "Buy-Till-You-Die" class of RFM models
that the `lifetimes <https://lifetimes.readthedocs.io/en/latest/#>`_ package is based on. A major pitfall of these models is the
rigid assumptions about distributions of hyperparameters, as well as the lack of
granularity of analysis. Wise_model aims to address these issues with a novel
implementation of recursive neural networks.

The Neural Lifetimes is a way to easily train a neural network on your data (for more
information on the neural net architecture, see `an overview of the model <high_level_overview.rst#Model>`_). Once you
have trained a model, you can use the `Inference package <neural_lifetimes/inference>`_ to predict customer
actions, extend existing customer sequences, or simulate entirely new sequences.

The Neural Lifetimes is based on a few assumptions:

1. Customers interact with the firm when they are “alive” between each timestep.
2. At each timestep, there is a probability of the customer "dying". This probability is sampled from the latent space.

Applications
~~~~~~~~~~~~

Common applications include:

- Predicting customers transactions (alive = actively purchasing,
  dead = not buying anymore).
- Clustering your customers based on their demographic information and
  their behaviour.
- Predicting the churn probability your customers given their purchasing behaviour


Specific Examples
~~~~~~~~~~~~~~~~~

For some examples of what this library can do, see :doc:`Quickstart <_notebook_BTYD_visualisation>`.

Installation
------------

::

   pip install neural-lifetimes

Documentation and tutorials
---------------------------

:doc:`Official documentation <index>`

Questions? Comments? Requests?
------------------------------

Please create an issue in the `this
repository <https://github.com/transferwise/neural-lifetimes>`__.

.. Use the actual url
