import subprocess
import os
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class GitInformationLogger(pl.Callback):
    def __init__(self, prefix="git/", verbose: bool = True) -> None:
        """Collects information on the state of the git repository for reproducibility.

        Args:
            prefix (str, optional): The prefix used for logging the contained information. Defaults to "git/".
            verbose (bool, optional): When true, warns when running from a dirty repository. Defaults to True.
        """
        super().__init__()
        self.prefix = prefix
        self.verbose = verbose

        # check whether run started in git repo:
        out = subprocess.run(["git", "status", "-s"], stdout=subprocess.PIPE)
        if out.stdout.startswith("fatal: not a git repository"):
            raise SystemError(
                "The entrypoint for the script is not inside a git repository. Consider running `git init` "
                + f"in you shell or remove the {self.__class__.__name__} callback."
            )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log(trainer, pl_module)
        return super().on_fit_start(trainer, pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log(trainer, pl_module)
        return super().on_test_start(trainer, pl_module)

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log(trainer, pl_module)
        return super().on_predict_start(trainer, pl_module)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log(trainer, pl_module)
        return super().on_validation_start(trainer, pl_module)

    def _log(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        git_information = self.data_dict()
        if isinstance(pl_module.logger, TensorBoardLogger):
            raise AssertionError(
                "This callback does not support the ``TensorBoardLogger``, due to its inability to process multiple "
                + "calls to log hyperparameters."
            )
        else:
            pl_module.logger.log_hyperparams(git_information)

    def data_dict(self) -> Dict[str, Any]:
        """Collects git information including git hash, commit date, short hash, status, branch and hostname.

        Returns:
            Dict[str, Any]: Git repository information.
        """
        git_information = {
            "short_hash": get_git_short_hash(),
            "hash": get_git_hash(),
            "commit_date": get_git_commit_date(),
            "status": get_git_repository_status(),
            "branch": get_git_branch(),
            "host": get_hostname(),
        }
        git_information = {f"{self.prefix}{key}": _decode(value) for key, value in git_information.items()}
        self._print_git_status_warning(git_information[f"{self.prefix}status"])
        return git_information

    def _print_git_status_warning(self, git_status: str):
        if not self.verbose:
            return
        if git_status == "dirty":
            print("\nWARNING: The git repository contains untracked files or uncommited changes.\n")
        if git_status == "unknown":
            print(
                "\nWARNING: The git repository status could not be determined. The git repository may contain "
                + "untracked files or uncommited changes.\n"
            )


def get_git_hash():
    return subprocess.check_output(["git", "log", "-n", "1", "--pretty=tformat:%H"]).strip()


def get_git_short_hash():
    return subprocess.check_output(["git", "log", "-n", "1", "--pretty=tformat:%h"]).strip()


def get_git_commit_date():
    return subprocess.check_output(["git", "log", "-n", "1", "--pretty=tformat:%ci", "--date=short"]).strip()


def get_git_repository_status():
    out = subprocess.run(["git", "status", "-s"], stdout=subprocess.PIPE)
    return "dirty" if out.stdout else "clean"


def get_git_branch():
    return subprocess.check_output(["git", "branch", "--show-current"]).strip()


def get_hostname():
    return os.uname()[1]


def _decode(x):
    try:
        x = x.decode("utf-8")
    except AttributeError:
        pass
    return x
