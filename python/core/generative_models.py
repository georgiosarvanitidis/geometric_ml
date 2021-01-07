import torch
import torch.nn as nn


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

