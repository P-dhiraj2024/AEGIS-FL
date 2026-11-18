"""AEGIS-FL detector state and one round of Algorithm 1 at the UAV edge.

Per round t (all UAVs share the projection seed and the detector weights distributed by the
cloud, so scoring at the U UAVs is the same computation as scoring all updates at once):
  1. reference: the cloud trains the global model on half A of its clean root set (update g_c)
     and on `n_ref` non-IID shards of half B (reference updates; local-step budgets follow the
     IoT / smartphone mix).  c = sketch(g_c).
  2. sketch every received update (Eq. sketch), centre it (s~ = s - c) and append the per-layer
     log-norm ratios to the reference median (norm_features).
  3. four nonconformity scores, each normalised by this round's reference scores:
       vae        Eq. (score) of the VAE on the feature vector (ratio to the reference median)
       norm       largest per-layer log-norm excess, w.r.t. references with the same step budget
       align      misalignment -mean_l cos(dw_l, g_c,l) with the trusted direction
       collusion  max cosine between the sketch and any other received sketch
     conformal p-values against FIFO buffers of clean reference scores (Eq. conformal);
     reject if min_j p_j <= alpha_t / J (Bonferroni, J = number of tests), so that a benign update
     exchangeable with the references is rejected with probability <= alpha_t.
  4. adaptive alpha (ACI) from the error on trusted clients (rho >= rho_trust) + references;
     reputation (Eq. rep) drives scheduling.
  5. VAE buffer <- reference sketches + accepted sketches of trusted clients; every K rounds
     retrain VAE + generator (Eq. gan).
  6. accepted updates are clipped to clip_mult x median reference norm, masked and summed at the
     cloud with RDP noise (fl/cloud.py).
Design history (pilots on CICIoT2023 / F-MNIST, seed 0, README): Eq. (sketch) + VAE alone was
nearly blind (late-round benign updates are almost orthogonal and normalisation removes norms);
the VAE scores identical colluding LIE / Min-Max updates as *more* normal than benign ones; a
calibration buffer fed by accepted clients was inflated by attackers that got accepted.
Ablations: no_norm (Eq. sketch + VAE only: no norm coordinates, no norm test), no_normtest,
no_align, no_collusion, no_vae (S = ||s~||^2), no_gan, no_conformal (thresholds frozen after
round 1, no adaptive alpha), no_reputation (reputation not used for scheduling).
"""
from __future__ import annotations

import numpy as np
import torch

from .conformal import OnlineConformal
from .gan_redteam import Generator, train_detector
from .reputation import Reputation
from .sketch import Sketcher
from .vae_detector import VAE

import os
_DEBUG = bool(os.environ.get("AEGIS_DEBUG"))


class Aegis:
    needs_root = True

    def __init__(self, fm, n_clients: int, dc: dict, seed: int, device, variant: str = "full"):
        self.dc, self.variant, self.L = dc, variant, len(fm.layers)
        self.sk = Sketcher(fm.layers, dc["sketch_dim_per_layer"], seed=10_000 + seed, device=device)
        self.norm_feat = bool(dc.get("norm_features", False)) and variant != "no_norm"
        self.layers = fm.layers
        self.dup_feat = bool(dc.get("dup_feature", False))
        dim = self.sk.dim + (self.L if self.norm_feat else 0) + (1 if self.dup_feat else 0)
        self.vae = VAE(dim, dc["vae_latent"]).to(device)
        self.gen = Generator(dim).to(device)
        self.opt_v = torch.optim.Adam(self.vae.parameters(), lr=dc.get("vae_lr", 1e-3))
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=dc.get("gen_lr", 1e-3))
        self.cp = OnlineConformal(dc["conformal_alpha"], dc["calib_buffer"], dc.get("aci_eps", 0.005),
                                  adaptive=variant != "no_conformal")      # holds alpha_t (ACI)
        self.tests = ["vae"] + [k for k in ("norm", "align", "collusion")
                                if dc.get(f"{k}_test", False) and variant != f"no_{k}" and not (k == "norm" and variant == "no_normtest")]
        self.cps = {k: OnlineConformal(dc["conformal_alpha"], dc["calib_buffer"]) for k in self.tests}
        self.rep = Reputation(n_clients, dc["reputation_beta"], dc["reputation_min"], dc.get("reputation_init", 0.5),
                              dc.get("probe_prob", 0.02), dc.get("trust_min", 0.55))
        self.buf: list[torch.Tensor] = []
        self.buf_max = dc.get("vae_buffer", 2000)
        self.tg = torch.Generator(device=device)
        self.tg.manual_seed(20_000 + seed)
        self.ready = False
        self.frozen_tau = None
        self.last_train = 0
        self.use_gan = variant != "no_gan"
        self.m_t = 1.0
        self.dump: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def eligible(self, rng):
        if self.variant == "no_reputation":
            return None
        return self.rep.eligible(rng)

    def _score(self, s):
        if self.variant == "no_vae":
            return s.pow(2).sum(1)
        with torch.no_grad():
            return self.vae.score(s, self.dc["score_weight_omega"])

    def _train(self, steps):
        if self.variant == "no_vae":
            return
        data = torch.cat(self.buf)[-self.buf_max:]
        per = bool(self.dc.get("per_coord_scale", False))
        if per or not self.ready:
            self.vae.fit_input_stats(data, per)
        tau = self.cps["vae"].threshold_at(self.cp.alpha_t) * self.m_t if self.ready else float("inf")
        s_cap = 2 * tau if np.isfinite(tau) else 1e3
        train_detector(self.vae, self.gen, self.opt_v, self.opt_g, data, self.c, self.L, steps,
                       self.dc.get("gan_nu", 0.1), s_cap, self.use_gan, self.tg)
        self.vae.fit_latent_stats(data)

    # ------------------------------------------------------------------ round
    def step(self, t: int, U: torch.Tensor, clients: np.ndarray, U_ref: torch.Tensor, g_c: torch.Tensor,
             truth: np.ndarray | None = None, steps: np.ndarray | None = None, steps_ref: np.ndarray | None = None):
        """truth (malicious flags) is used only for the AEGIS_DEBUG diagnostics, never for decisions."""
        self.c = self.sk(g_c[None])[0]
        Sr, Sr_ref = self.sk(U), self.sk(U_ref)
        S, S_ref = Sr - self.c, Sr_ref - self.c
        if self.norm_feat:                                   # per-layer log-norm relative to the references
            ln = lambda X: torch.stack([X[:, a:b].norm(dim=1).clamp(min=1e-12).log() for a, b in self.layers], 1)
            nu = ln(U_ref).median(0).values
            S = torch.cat([S, self.dc.get("norm_scale", 1.0) * (ln(U) - nu)], 1)
            S_ref = torch.cat([S_ref, self.dc.get("norm_scale", 1.0) * (ln(U_ref) - nu)], 1)
        if self.dup_feat:                                    # collusion: max cosine to another received sketch
            nrm = lambda X: torch.nn.functional.normalize(X, dim=1)
            A, B = nrm(Sr), nrm(Sr_ref)
            cc = A @ A.T
            cc.fill_diagonal_(-1.0)
            S = torch.cat([S, self.dc.get("dup_scale", 1.0) * cc.max(1).values[:, None]], 1)
            S_ref = torch.cat([S_ref, self.dc.get("dup_scale", 1.0) * (B @ A.T).max(1).values[:, None]], 1)
        first = not self.ready
        if first:                                            # offline pre-training on the references
            self.buf.append(S_ref)
            self._train(self.dc.get("pretrain_steps", 300))
        # ---- nonconformity scores (all: larger = more anomalous), normalised by this round's references
        ref_sc = self._score(S_ref).cpu().numpy()
        self.m_t = float(np.median(ref_sc))                  # VAE score: ratio to the reference median (drift)
        scores = {"vae": self._score(S).cpu().numpy() / self.m_t}
        ref_scores = {"vae": ref_sc / self.m_t}
        if "norm" in self.tests:                             # largest per-layer log-norm excess over the references
            lnn = lambda X: torch.stack([X[:, a:b].norm(dim=1).clamp(min=1e-12).log() for a, b in self.layers], 1)
            LR, LU = lnn(U_ref), lnn(U)
            if steps is not None and steps_ref is not None and self.dc.get("device_aware_norm", True):
                # update norms grow with the local-step budget (IoT 5 vs smartphone 10): compare each
                # client with the references that ran the same budget
                NR, NU = torch.zeros_like(LR), torch.zeros_like(LU)
                for st in np.unique(np.concatenate([steps, steps_ref])):
                    mr = torch.as_tensor(steps_ref == st, device=U.device)
                    base = LR[mr].median(0).values if bool(mr.any()) else LR.median(0).values
                    NR[mr] = LR[mr] - base
                    mu_ = torch.as_tensor(steps == st, device=U.device)
                    NU[mu_] = LU[mu_] - base
            else:
                nu = LR.median(0).values
                NR, NU = LR - nu, LU - nu
            scores["norm"], ref_scores["norm"] = NU.max(1).values.cpu().numpy(), NR.max(1).values.cpu().numpy()
        if "align" in self.tests:                            # misalignment with the trusted direction g_c
            mis = lambda X: -torch.stack([torch.nn.functional.cosine_similarity(X[:, a:b], g_c[None, a:b], dim=1)
                                          for a, b in self.layers], 1).mean(1).cpu().numpy()
            r = mis(U_ref)
            scores["align"], ref_scores["align"] = mis(U) - np.median(r), r - np.median(r)
        if "collusion" in self.tests:                        # max sketch cosine to another received update
            A, B = torch.nn.functional.normalize(Sr, dim=1), torch.nn.functional.normalize(Sr_ref, dim=1)
            cc = A @ A.T
            cc.fill_diagonal_(-1.0)
            r = (B @ A.T).max(1).values.cpu().numpy()
            scores["collusion"], ref_scores["collusion"] = cc.max(1).values.cpu().numpy() - np.median(r), r - np.median(r)
        J = len(self.tests)
        if first:
            for k in self.tests:
                self.cps[k].add(ref_scores[k])
            self.ready, self.last_train = True, t
            if self.variant == "no_conformal":
                self.frozen = {k: self.cps[k].threshold_at(self.cp.alpha / J) for k in self.tests}
        if self.variant == "no_conformal":
            accept = np.all([scores[k] <= self.frozen[k] for k in self.tests], 0)
        else:
            pmin = np.min([self.cps[k].pvalue(scores[k]) for k in self.tests], 0)
            accept = pmin > self.cp.alpha_t / J
        sc = scores["vae"]
        tau = self.cps["vae"].threshold_at(self.cp.alpha_t / J)
        if self.dc.get("dump_scores", False) and truth is not None:      # diagnostics only (figure)
            self.dump.append(dict(t=t, mal=truth.astype(bool), accept=accept,
                                  thr={k: self.cps[k].threshold_at(self.cp.alpha_t / J) for k in self.tests},
                                  **{k: scores[k] for k in self.tests}, **{f"ref_{k}": ref_scores[k] for k in self.tests}))
        if _DEBUG and truth is not None and (t <= 5 or t % 25 == 0):
            m = truth.astype(bool)
            q = lambda x: np.round(np.quantile(x, [.1, .5, .9]), 2) if len(x) else "-"
            thr = {k: round(self.cps[k].threshold_at(self.cp.alpha_t / J), 2) for k in self.tests}
            print(f"  DIAG t={t} thr {thr} rej ben {(~accept[~m]).mean():.2f} mal {(~accept[m]).mean() if m.any() else 0:.2f} | "
                  + " | ".join(f"{k} ben {q(scores[k][~m])} mal {q(scores[k][m])} ref {q(ref_scores[k])}" for k in self.tests),
                  flush=True)
        if _DEBUG:
            print(f"  dbg t={t} alpha_t={self.cp.alpha_t:.4f} accept {accept.mean():.2f} "
                  + " ".join(f"{k}:{np.quantile(scores[k], [.1, .5, .9]).round(2)}" for k in self.tests), flush=True)
        trusted = self.rep.trusted(clients)
        # calibration feedback: trusted clients (accepted or not) + this round's fresh references
        refs_only = self.dc.get("calib_source", "trusted") == "refs"
        if not first:
            fb = {k: (ref_scores[k] if refs_only else np.concatenate([scores[k][trusted], ref_scores[k]]))
                  for k in self.tests}
            if self.variant != "no_conformal":
                # ACI feedback: error on the reference scores, or on trusted clients + references
                fa = fb if self.dc.get("aci_feedback", "calib") == "calib" else \
                    {k: np.concatenate([scores[k][trusted], ref_scores[k]]) for k in self.tests}
                pf = np.min([self.cps[k].pvalue(fa[k]) for k in self.tests], 0)
                self.cp.update(float((pf <= self.cp.alpha_t / J).mean()))
            for k in self.tests:
                self.cps[k].add(fb[k])
        elif trusted.any() and not refs_only:
            for k in self.tests:
                self.cps[k].add(scores[k][trusted])
        self.rep.update(clients, accept)
        keep = torch.as_tensor(trusted & accept, device=U.device)
        self.buf.append(S[keep] if first else torch.cat([S_ref, S[keep]]))
        tot = sum(len(b) for b in self.buf)
        while tot - len(self.buf[0]) >= self.buf_max:
            tot -= len(self.buf.pop(0))
        if t - self.last_train >= self.dc["gan_retrain_every"]:
            self._train(self.dc.get("retrain_steps", 100))
            self.last_train = t
        return accept, sc, tau, int(keep.sum())
