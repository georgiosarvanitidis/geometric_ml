import torch
import torch.nn as nn
import numpy as np

class VAE_model(nn.Module):

    def __init__(self, d, D, H, activFun):
        super(VAE_model, self).__init__()

        # The VAE components
        self.enc = nn.Sequential(
            nn.Linear(D, H),
            activFun,
            nn.Linear(H, H),
            activFun
        )

        self.mu_enc = nn.Sequential(
            self.enc,
            nn.Linear(H, d)
        )

        self.log_var_enc = nn.Sequential(
            self.enc,
            nn.Linear(H, 1)
        )

        self.dec = nn.Sequential(
            nn.Linear(d, H),
            activFun,
            nn.Linear(H, H),
            activFun
        )

        self.mu_dec = nn.Sequential(
            self.dec,
            nn.Linear(H, D)
        )

        self.log_var_dec = nn.Sequential(
            self.dec,
            nn.Linear(H, D)
        )

    def encode(self, x):
        return self.mu_enc(x), self.log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z), self.log_var_dec(z)

    # Note: Define the model
    @staticmethod
    def reparametrization_trick(mu, log_var):
        epsilon = torch.randn_like(mu)  # the Gaussian random noise
        return mu + torch.exp(0.5 * log_var) * epsilon

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z_rep = self.reparametrization_trick(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z_rep)

        return mu_x, log_var_x, z_rep, mu_z, log_var_z


# Computes the objective function of the VAE
def VAE_loss(x, mu_x, log_var_x, mu_z, log_var_z, r=1.0):
    D = mu_x.shape[1]
    d = mu_z.shape[1]

    # The decoder and encoder is common for all models
    if log_var_x.shape[1] == 1:
        P_X_Z = + 0.5 * (D * log_var_x + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()
    else:
        P_X_Z = + 0.5 * (log_var_x.sum(dim=1, keepdim=True)
                         + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()

    if log_var_z.shape[1] == 1:
        Q_Z_X = - 0.5 * (d * log_var_z).mean()
    else:
        Q_Z_X = - 0.5 * log_var_z.sum(dim=1, keepdim=True).mean()

    if log_var_z.shape[1] == 1:
        P_Z = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + d * log_var_z.exp()).mean()
    else:
        P_Z = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + log_var_z.exp().sum(dim=1, keepdim=True)).mean()

    return P_X_Z + r * Q_Z_X + r * P_Z


class VariationalAutoEncoder_GELS(nn.Module):

    def __init__(self, D=None, d=None, H=None, activFun=None, A_init=None, b_init=None):
        super(VariationalAutoEncoder_GELS, self).__init__()

        # Encoder
        self.mu_enc = nn.Sequential(
            nn.Linear(D, 2 * H),
            activFun,
            nn.Linear(2 * H, H),
            activFun,
            nn.Linear(H, d)
        )

        self.log_var_enc = nn.Sequential(
            nn.Linear(D, 2 * H),
            activFun,
            nn.Linear(2 * H, H),
            activFun,
            nn.Linear(H, d)
        )

        # Decoder
        self.mu_dec = nn.Sequential(
            nn.Linear(d, H),
            activFun,
            nn.Linear(H, 2 * H),
            activFun,
            nn.Linear(2 * H, D)
        )

        self.log_var_dec = nn.Sequential(
            nn.Linear(d, H),
            activFun,
            nn.Linear(H, 2 * H),
            activFun,
            nn.Linear(2 * H, D)
        )

        # The linear extrapolation part
        self.mu_dec_resNet = nn.Linear(d, D)  # The res-net of the decoder

        # Initialize the linear part of the decoder
        if A_init is not None:
            with torch.no_grad():
                self.mu_dec_resNet.weight.copy_(torch.from_numpy(A_init.astype(np.float32)))
                self.mu_dec_resNet.weight.requires_grad = False
                self.mu_dec_resNet.bias.copy_(torch.from_numpy(b_init.flatten().astype(np.float32)))
                self.mu_dec_resNet.bias.requires_grad = False
        else:
            with torch.no_grad():
                self.mu_dec_resNet.weight.copy_(torch.eye(D, d))
                self.mu_dec_resNet.weight.requires_grad = False
                self.mu_dec_resNet.bias.copy_(torch.zeros(D))
                self.mu_dec_resNet.bias.requires_grad = False

    @staticmethod
    # The reparametrization trick
    def reparametrization_trick(mu_enc, log_var_enc):
        epsilon = torch.randn_like(mu_enc)  # the Gaussian random noise
        return mu_enc + torch.exp(0.5 * log_var_enc) * epsilon

    def encode(self, x):
        return self.mu_enc(x), self.log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z) + self.mu_dec_resNet(z), self.log_var_dec(z)

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)  # Points in the latent space
        z_rep = self.reparametrization_trick(mu_z, log_var_z)  # Samples from z ~ q(z | x)
        mu_x, log_var_x = self.decode(z_rep)  # The decoded samples
        return mu_x, log_var_x, z_rep, mu_z, log_var_z


# The VAE loss
def VAE_loss_GELS(x, mu_x, log_var_x, mu_z, log_var_z, anneal_param):
    DATA_FIT = + 0.5 * (log_var_x.sum(dim=1, keepdim=True)
                     + (((x - mu_x) ** 2) / log_var_x.exp()).sum(dim=1, keepdim=True)).mean()

    KLD = + 0.5 * ((mu_z ** 2).sum(dim=1, keepdim=True) + log_var_z.exp().sum(dim=1, keepdim=True)).mean() \
          - 0.5 * log_var_z.sum(dim=1, keepdim=True).mean()

    return DATA_FIT + anneal_param * KLD
