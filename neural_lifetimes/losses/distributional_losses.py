from typing import Callable, Dict, Sequence

import numpy as np
import torch
import torch.distributions as d
import torch.nn.functional as F
from torch import nn

# different kinds of losses:

# Time to next transaction: exponential, halfNormal prior
# next amount: logNormal, halfnormal priors for mean and std
# next discrete label: good ol' softmax and entropy


class CompositeLoss(nn.Module):
    def __init__(self, dl: Dict[str, nn.Module], preprocess: Callable = lambda x: x):
        super().__init__()
        self.losses = nn.ModuleDict(dl)
        self.preprocess = preprocess

    def forward(self, y_pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        tgt, yp = self.preprocess(target, y_pred)
        losses = {name: loss(yp[name], tgt[name]) for name, loss in self.losses.items()}
        return sum(list(losses.values())), losses


class SumLoss(nn.Module):
    def __init__(self, s: Sequence[nn.Module]):
        super().__init__()
        self.losses = nn.ModuleDict(s)

    def forward(self, y_pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        losses = {name: loss(y_pred, target) for name, loss in self.losses.items()}
        loss_sum = sum([loss[0] for loss in losses.values()])
        loss_dict = {name: loss[0] for name, loss in losses.items()}
        [loss_dict.update(loss[1]) for loss in losses.values()]
        # loss_dict = dict(ChainMap([loss[1] for loss in losses.values()]))
        # [loss_dict.update(d) for name, value
        return loss_sum, loss_dict


class CategoricalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor):
        ll = F.nll_loss(y_pred, target)
        assert not torch.isnan(ll), "Got a NaN loss"
        assert ll not in [-np.inf, np.inf], "Loss not finite!"

        return ll


class ExponentialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor):
        eps = 0.00001
        dist = d.exponential.Exponential(1.0 / y_pred)
        ll = -torch.diagonal(dist.log_prob(target + eps)).mean()
        assert not torch.isnan(ll), "Got a NaN loss"
        assert ll not in [-np.inf, np.inf], "Loss not finite!"

        return ll


class LogNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor):
        dist = d.log_normal.LogNormal(y_pred[:, 0], y_pred[:, 1])
        ll = -dist.log_prob(target + 0.01).mean()
        assert not torch.isnan(ll), "Got a NaN loss"
        assert ll not in [-np.inf, np.inf], "Loss not finite!"

        return ll


class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor):
        target[target.isnan()] = 0
        dist = d.normal.Normal(y_pred[:, 0], torch.exp(torch.clamp(y_pred[:, 1], min=-30, max=50)))
        ll = -dist.log_prob(target + 0.01).mean()
        assert not torch.isnan(ll), "Got a NaN loss"
        assert ll not in [-np.inf, np.inf], "Loss not finite!"

        return ll


class ChurnLoss(nn.Module):
    def __init__(self, dt_dist_gen, scale_by_seq=False):
        super().__init__()
        self.dt_dist_gen = dt_dist_gen
        self.scale_by_seq = scale_by_seq

    def forward(self, y_pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        # let's slice it up into sub-sequences corresponding to individual users
        seq_inds = zip(target["offsets"][:-1], target["offsets"][1:])

        tmp = {
            "dt": target["dt"],
            "tau": y_pred["next_dt"].view(-1),
            "p": y_pred["p_churn"].view(-1),
        }

        # clip the first item because its dt is always zero (= t+0 - t_{start token})
        seqs = []
        for s, e in seq_inds:
            seqs.append(
                {  # the first two are zeros anyway
                    "dt": tmp["dt"][(s + 2) : e],
                    "tau": tmp["tau"][(s + 1) : e],
                    "p": tmp["p"][(s + 1) : e],
                }
            )

        assert len(seqs) == len(target["t_to_now"]), "Sequence length misalignment!"

        loss = torch.tensor(0.0, device=y_pred["p_churn"].device)
        eps = 0.00001

        for seq, t_to_now in zip(seqs, target["t_to_now"]):
            this_loss = torch.tensor(0.0, device=y_pred["p_churn"].device)
            if len(seq["dt"]) > 0:
                # the loss from them not having churned earlier, in spite of the probabilities
                this_loss -= torch.log(1 - seq["p"][:-1] + eps).sum()

                dist = self.dt_dist_gen.distribution(seq["tau"][:-1])
                # dist = d.exponential.Exponential(1.0 / seq["tau"][1:-1])
                # the loss of having just that dt
                this_loss -= dist.log_prob(seq["dt"] + eps).sum()

            assert not torch.isnan(this_loss), "Got a NaN loss"
            # assert loss not in [-np.inf, np.inf], "Loss not finite!" # DUPE
            assert this_loss not in [-np.inf, np.inf], "Churn loss not finite!"

            last_p = seq["p"][-1]
            last_dist = self.dt_dist_gen.distribution(seq["tau"][-1])
            # last_exp_cdf = torch.clamp(
            #     torch.exp(-t_to_now / last_tau), min=1e-7, max=3.4e38
            # )  # 3.4e38 < max(torch.float32)
            this_loss -= torch.log((1 - last_p) * (1 - last_dist.cdf(t_to_now + eps)) + last_p + eps)
            assert this_loss not in [-np.inf, np.inf], "Churn loss not finite!"
            assert not torch.isnan(this_loss), "Got a NaN loss"
            if self.scale_by_seq:
                this_loss /= len(seq["p"])

            loss += this_loss
        if self.scale_by_seq:
            final_loss = loss / len(seqs)
        else:
            final_loss = loss / len(target["t"])

        assert final_loss not in [-np.inf, np.inf], "Churn loss not finite!"
        assert not torch.isnan(final_loss), "Got a NaN loss"

        return final_loss, {"churn": final_loss}


class TauLoss(nn.Module):  # This is purely cosmetic, for tensorboard purposes
    # This is the new Tau Loss function (there are issues with target['next_dt'] vs target['dt']).
    # The aim is to be able to normalise per person and not per data point.
    def forward(self, y_pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        # let's slice it up into sub-sequences corresponding to individual users
        seq_inds = zip(target["offsets"][:-1], target["offsets"][1:])
        tmp = {
            "nextdt": target["next_dt"],
            "dt": target["dt"],
            "tau": y_pred["next_dt"].detach(),
        }
        seqs = [{k: v[s:e] for k, v in tmp.items()} for s, e in seq_inds]
        eps = 0.00001
        loss = 0.0
        for seq in seqs:
            # the loss from them not having churned earlier, in spite of the probabilities
            eps = 0.00001
            dist = d.exponential.Exponential(1.0 / seq["tau"][1:-1])
            loss -= torch.diagonal(dist.log_prob(seq["dt"][1:-1] + eps)).sum()
            assert not torch.isnan(loss), "Got a NaN loss"
            assert loss not in [-np.inf, np.inf], "Loss not finite!"

        final_loss = loss / target["offsets"].shape[0]
        return final_loss, {"dt": final_loss}
