import numpy as np

import torch
import torch.nn as nn


class VariationalEncoderDecoderLoss(nn.Module):
    # Adds a variational term to any kind of final loss
    # Assumes final loss is a log likelihood

    def __init__(self, fit_loss: nn.Module, reg_weight=1):
        super().__init__()
        self.final_loss = fit_loss
        self.reg_weight = reg_weight

    def forward(self, model_out, target_x) -> torch.Tensor:  # changed order to follow pytorch convention

        fit_loss, losses_dict = self.final_loss(model_out, target_x)
        mu = model_out["mu"]
        std = model_out["std"]

        if self.reg_weight is not None:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD_element = 1 + torch.log(std * std) - mu * mu - std * std
            KLD = -0.5 * torch.mean(KLD_element)
            my_loss = fit_loss + self.reg_weight * KLD
        else:  # the case when we do no sampling
            my_loss = fit_loss
            KLD = 0.0

        losses_dict["kl_div"] = KLD
        losses_dict["model_fit"] = fit_loss
        losses_dict["total"] = my_loss
        losses_dict = {f"loss/{name}": loss for name, loss in losses_dict.items()}

        assert my_loss not in [-np.inf, np.inf], "Loss not finite!"
        assert not torch.isnan(my_loss), "Got a NaN loss"

        assert sum(losses_dict.values()) not in [-np.inf, np.inf], "Loss not finite!"
        assert not torch.isnan(sum(losses_dict.values())), "Got a NaN loss"

        return my_loss, losses_dict


# class ELBOLoss(nn.Module):
#     def __init__(self, fit_loss: nn.Module, reg_weight) -> None:
#         super().__init__()
#         self.fit_loss = fit_loss
#         self.reg_weight = reg_weight

#     def forward(self, pred: torch.Tensor, target: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
#         fit_loss, losses_dict = self.final_loss(model_out, target_x)
#         mu = model_out["mu"]
#         std = model_out["std"]
