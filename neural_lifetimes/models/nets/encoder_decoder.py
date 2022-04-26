import torch
from torch import nn
from torch.distributions.normal import Normal


class VariationalEncoderDecoder(nn.Module):
    """
    An implementation of variational encoder and decoder.

    Args:
        encoder(nn.Module): A model mapping batches of source domain to batches of vectors (batch x input_size)
        decoder(nn.Module): Model mapping latent z (batch x z_size) to  target domain
        sample_z(bool): Whether to sample z = N(mu, std) or just take z=mu. Defaults to ``True``.
        epsilon_std(float): Scaling factor for sampling, low values help convergence. Defaults to ``1.0``.

    Note:
        See https://github.com/mkusner/grammarVAE/issues/7
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sample_z: bool = True,
        epsilon_std: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_size = self.decoder.input_shape[1]
        self.sample_z = sample_z
        self.epsilon_std = epsilon_std

        self.fc_mu = nn.Linear(self.encoder.output_shape[-1], self.z_size)
        self.fc_log_var = nn.Linear(self.encoder.output_shape[-1], self.z_size)

    def forward(self, x):
        """
        Encoder-decoder forward pass.

        Args:
            x: A batch of source domain data (batch x input_size)

        Returns:
            The data after being passed through the encoder and decoder.
        """
        enc_out = self.encoder(x)
        mu = self.fc_mu(enc_out)
        log_var = self.fc_log_var(enc_out)

        eps = 0.0001
        std = torch.square(log_var) + eps  # no longer log, why?

        if self.sample_z:
            z = Normal(mu, self.epsilon_std * std).rsample()
        else:
            z = mu

        output = self.decoder(z)
        output["mu"] = mu
        output["std"] = std
        output["sampled_z"] = z

        return output

    def load(self, weights_file):
        print("Trying to load model parameters from ", weights_file)
        self.load_state_dict(torch.load(weights_file))
        self.eval()
        print("Success!")
