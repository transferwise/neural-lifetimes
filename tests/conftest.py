import pytest
from pytorch_lightning.utilities import seed


@pytest.fixture(autouse=True)
def random_seed():
    """
    For reproducibility, set the random seed everywhere.

    PyTorch Lightning's seed.seed_everything sets the random seen in torch, numpy and
    python.random. Pandas uses the random state from numpy.
    """
    seed.seed_everything(0)
