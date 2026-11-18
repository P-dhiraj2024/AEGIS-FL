"""FLTrust (Cao et al., NDSS 2021): trust score ReLU(cos(g_i, g0)) w.r.t. the server update g0
on the clean root set (the same root set AEGIS-FL uses for its reference updates)."""
import torch


class FLTrust:
    needs_root = True

    def aggregate(self, U, ctx):
        g0 = ctx["g0"]
        ts = torch.relu(torch.nn.functional.cosine_similarity(U, g0[None], dim=1))
        if ts.sum() <= 0:
            return torch.zeros_like(g0), torch.ones(len(U), dtype=torch.bool, device=U.device)
        scaled = U * (g0.norm() / U.norm(dim=1).clamp(min=1e-12))[:, None]
        return (ts[:, None] * scaled).sum(0) / ts.sum(), ts <= 0
