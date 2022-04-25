# Neural Lifetimes
#TODO Insert Logo

![test](https://github.com/transferwise/neural_lifetimes_clean/actions/workflows/test.yml/badge.svg)
![lint](https://github.com/transferwise/neural_lifetimes_clean/actions/workflows/lint.yml/badge.svg)
![format](https://github.com/transferwise/neural_lifetimes_clean/actions/workflows/format.yml/badge.svg)
![docs](https://github.com/transferwise/neural_lifetimes_clean/actions/workflows/docs.yml/badge.svg)
![pypi](https://img.shields.io/pypi/v/neural-lifetimes)

# Introduction

The Neural Lifetimes package is an open-source lightweight framework based on [PyTorch](https://pytorch.org/) and [PyTorch-Lightning](https://www.pytorchlightning.ai/) to conduct modern lifetimes analysis based on neural network models. This package provides both flexibility and simplicity:
- Users can use the simple interface to load their own data and train good models _out-of-the-box_ with very few lines of code.
- The modular design of this package enables users to selectively pick individual tools.

Possible usage of Neural Lifetimes is

- Predicting customer transactions
- Calculating Expected Customer Lifetime Values
- Obtain Customer Embeddings
- TODO add more

# Features

## Simple Interface

You can run your own dataset with a few lines of code:

## Data

We introduce a set of tools to

- Load data in batches from database
- Handle sequential data
- Load data from interfaces such as Pandas, Clickhouse, Postgres, VAEX and more

We further provide a simulated dataset based on the `BTYD` model for exploring this package and we provide tutorials to understand the mechanics of this model.

## Models

We provide a simple `GRU`-based model that embeds any data and predicts sequences of transactions.


## Model Inference

The class `inference.ModelInference` allows to simulate sequences from scratch or extend sequences from a model artifact.
A sequence is simulated/extended iteratively by adding one event at the end of the sequence each time.
To simulate an event, the current sequence is used as the model input and the distributions outputted by the model are
used to sample the next event. The sampled event is added to the sequence and the resulting sequence is used as an input
in the following iteration. The process ends if a sequence reaches the `end_date` or if the
customer churns.

To initialize the `ModelInference` class needs, you need to give the filepath of a trained model artifact:
```
inference = ModelInference(
    model_filename = "/neural_lifetimes_clean/data/logs/eventsprofiles/btyd/version_1/epoch=0-step=1-val_loss_total=1.0.ckpt"
)
```

`ModelInference` has two main methods:

- `simulate_sequences`: simulates `n` sequences from scratch. The sequences start with an event randomly sampled between
  `start_date` and  `start_date_limit`. The sequences of events are build by sampling
  from the model distribution ouputs. The sequence is initialized with a Starting Token event.
  A sequence will end when if either the user churns or if an event happens after the
  `end_date`.

```
simulate_sequences = inference.simulate_sequences(
    n = 10,
    start_date = datetime.datetime(2021, 1, 1, 0, 0, 0),
    start_date_limit = datetime.datetime(2021, 2, 1, 0, 0, 0),
    end_date = datetime.datetime(2021, 4, 1, 0, 0, 0),
    start_token_discr = 'StartToken',
    start_token_cont = 0
)
```

- `extend_sequence`: takes a `ml_utils.torch.sequence_loader.SequenceLoader` loader and the start and end date of the
  simulation. The method processes the loader in batches. The `start_date` must be after any event in any sequence. Customers might have already churned after their last event
  so we first need to infer the churn status of the customers. To infer the churn status, we input a sequence into the model
  and sample from the output distributions. If the churn status after the last event is True or the next event would have
  happened before `start_date` we infer that that customer has churned.
  For all the customer sequence that haven't churned we extend the sequences as in `simulate_sequences`.
```
raw_data, extended_seq = inference.extend_sequence(
    loader,
    start_date = datetime.datetime(2021, 1, 1, 0, 0, 0),
    end_date = datetime.datetime(2021, 4, 1, 0, 0, 0),
    return_input = True
)
```

The `extend_sequence` method can return also the original sequences if `return_input = True`.
`extended_seq` contains list of dicts where each dict is a processed batch. Each dict has two keys: 'extended_sequences' and 'inferred_churn'.
'extended_sequences' contains the extended sequences that were inferred NOT to have churned.
'inferred_churn' contains the sequences that were inferred to have churned.


# Documentation

The documentation for this repository is available at

[TODO Add Link]()


# Install

You may install the package from [PyPI](https://pypi.org/project/neural-lifetimes/):

```bash
pip install neural-lifetimes
```

Alternatively, you may install from git to get access to the latest commits:

```bash
pip install git+https://github.com/transferwise/neural-lifetimes
```

# Getting started

In the documentation there is a tutorial on getting started.

[TODO add link]()

#TODO add google colab notebook to start


# Useful Resources

- Github: [Lifetimes Package](https://github.com/CamDavidsonPilon/lifetimes)
- Documentation: [PyTorch](https://pytorch.org/docs/stable/index.html/)
- Documentation: [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- Paper: [Fader et al. (2005), "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)


# Contribute

We welcome all contributions to this repository. Please read the [Contributing Guide](https://github.com/transferwise/neural_lifetimes_clean/blob/update-readme/CONTRIBUTING.md).

If you have any questions or comments please raise a Github issue.


-----------------
-----------------
# LEGACY
-----------------

## MOVE THIS TO "Getting Started" IN THE DOCS
A new virtual environment, `neural_lifetimes`, can be created and activated as follows:
```bash
             $ cd path/to/project/root
             $ conda create -n neural_lifetimes python=3.9
             $ conda activate neural_lifetimes
(neural_lifetimes) $ ...
```
First, install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you require. For example,

```bash
(neural_lifetimes) $ pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Second, the `requirements.txt` file specifies the dependencies you will need to install to use this repository.

```bash
(neural_lifetimes) $ pip install -r requirements.txt
```

## Run Script

The script to run the trainer is in `src/neural_lifetimes/train`:

`train_model.py` trains our own GRU-based encoder-decoder

The event model class is in `src/neural_lifetimes/model/model.py`.
