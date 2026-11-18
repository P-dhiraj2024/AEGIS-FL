"""GAN red team (Eq. gan).

min_{phi,theta} max_xi  E_z[ Psi(G_xi(z)) - nu S(G_xi(z)) ] - L_ELBO.

Generator.  G_xi maps noise z and the current reference sketch c to a malicious centred sketch
delta = G(z, c); its update direction per layer is c_l + delta_l.  Each coordinate is bounded
by `box` times the largest benign |deviation| of that coordinate, so generated attacks stay in
the plausible region that adaptive attacks target.

Psi surrogate.  The update behind a sketch is not available, so Psi uses the first-order loss
change: a client update u changes the global loss by about -eta <g, u>; JL projections preserve
inner products, so the layer-mean of 1 - cos(c_l + delta_l, c_l) (in [0, 2]) measures how far
the sketched direction turns away from the trusted descent direction c.

Detector side.  The detector maximises the ELBO of benign sketches and nu * min(S(fake), S_cap)
(the cap keeps the min-max game bounded; S_cap = 2 tau).  Schedule: offline pre-training on the
root reference sketches in round 1, then `steps` alternating updates every K rounds.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim: int, noise: int = 16, hidden: int = 128):
        super().__init__()
        self.noise = noise
        self.net = nn.Sequential(nn.Linear(noise + dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, z, c, box):
        return torch.tanh(self.net(torch.cat([z, c.expand(len(z), -1)], 1))) * box


def psi(delta: torch.Tensor, c: torch.Tensor, n_layers: int) -> torch.Tensor:
    """c: centre sketch [kL]; delta may carry extra (norm) coordinates after the first kL."""
    x = (c + delta[:, :c.numel()]).view(len(delta), n_layers, -1)
    cl = c.view(1, n_layers, -1)
    cos = (x * cl).sum(-1) / (x.norm(dim=-1) * cl.norm(dim=-1)).clamp(min=1e-12)
    return (1 - cos).mean(1)


def train_detector(vae, gen, opt_v, opt_g, benign: torch.Tensor, c: torch.Tensor, n_layers: int,
                   steps: int, nu: float, s_cap: float, use_gan: bool, gen_t: torch.Generator,
                   batch: int = 128, box_mult: float = 1.5):
    """Alternating min-max updates of Eq. (gan) on the benign buffer (centred sketches)."""
    vae.train()
    box = box_mult * benign.abs().amax(0).clamp(min=1e-6)
    if use_gan:                                   # generator conditioning: centre padded to the input dim
        c_in = torch.cat([c, torch.zeros(benign.shape[1] - c.numel(), device=c.device)])
    n = len(benign)
    for _ in range(steps):
        b = benign[torch.randint(0, n, (min(batch, n),), device=benign.device, generator=gen_t)]
        loss_v = vae.neg_elbo(b).mean()
        if use_gan:
            z = torch.randn(len(b), gen.noise, device=b.device, generator=gen_t)
            fake = gen(z, c_in, box).detach()
            loss_v = loss_v - nu * vae.score(fake).clamp(max=s_cap).mean()
        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()
        if use_gan:
            z = torch.randn(len(b), gen.noise, device=b.device, generator=gen_t)
            fake = gen(z, c_in, box)
            loss_g = -(psi(fake, c, n_layers) - nu * vae.score(fake).clamp(max=s_cap)).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
    vae.eval()
