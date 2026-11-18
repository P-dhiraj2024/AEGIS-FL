"""Main entry: one AEGIS-FL / baseline experiment over the IoT+mobile -> UAV -> cloud hierarchy.

    python -m src.run --config configs/default.yaml --seed 0 [--set data.dataset=cifar10 ...]
    python -m src.run --config configs/default.yaml --smoke      # 2 rounds, 20 clients

Per round (Algorithm 1): mobility + availability -> scheduling (reputation / bans) -> UAV
association, rates, latency, energy -> deadline drops stragglers -> batched local training
(data poisoning inside) -> model poisoning -> defence (AEGIS at the UAV edge + secure/DP cloud
aggregation, or a cloud-side baseline) -> global update.  Every `eval_every` rounds the model
is evaluated on the validation set (model selection / curves) and the test set (reported).
Results: results/runs/<exp>/<seed>/{config.yaml, history.json, summary.json, meta.json}.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from core.metrics.metrics import precision_recall_f1
from core.utils.config import apply_overrides, load_config
from core.utils.seed import set_seed
from core.utils.tracking import RunLogger

from .attacks import BACKDOORS, MODEL_ATTACKS, model_poison
from .attacks.adaptive import AdaptiveAttacker
from .attacks.neurotoxin import mask_from_global_update
from .baselines import build_baseline
from .data import load_fed, stamp_trigger
from .defense.aegis import Aegis
from .fl.client import BatchedTrainer, Poison, root_tables
from .fl.cloud import Cloud
from .fl.uav_edge import Network
from .models import FlatModel, build_model, count_macs

RESULTS = __import__("pathlib").Path(__file__).resolve().parents[1] / "results"


def exp_name(cfg) -> str:
    d, a = cfg["defense"], cfg["attack"]
    name = d["name"] + ("" if d["variant"] == "full" or d["name"] != "aegis" else f"-{d['variant']}")
    tag = f"{a['type']}_g{a['ratio']}" if a["type"] != "none" else "none"
    if a["type"] == "adaptive" and float(a["adaptive"].get("target_quantile", 0.9)) != 0.9:
        tag = f"adaptive-q{a['adaptive']['target_quantile']}_g{a['ratio']}"
    if d["name"] == "aegis" and d["conformal_alpha"] != 0.05:
        tag += f"_a{d['conformal_alpha']}"
    if math.isfinite(float(cfg["privacy"]["dp_epsilon"])):
        tag += f"_eps{cfg['privacy']['dp_epsilon']}"
    return f"{cfg['data']['dataset']}/{name}/{tag}"


@torch.no_grad()
def evaluate(fm, fd, w, x, y, backdoor: bool):
    pred = fm.predict(w, fd.to_input(x) if fd.kind == "image" else x)
    acc = float((pred == y).float().mean())
    _, _, f1 = precision_recall_f1(y.cpu().numpy(), pred.cpu().numpy(), fd.n_classes)
    out = dict(acc=acc, f1=f1)
    if backdoor:
        keep = y != fd.target_class
        xt = fd.to_input(x[keep]) if fd.kind == "image" else x[keep]
        pt = fm.predict(w, stamp_trigger(fd, xt, None))
        out["asr"] = float((pt == fd.target_class).float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--exp", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg.pop("sweep", None)
    cfg.pop("variants", None)
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.smoke:
        cfg.update(rounds=2, num_clients=20, clients_per_round=10, eval_every=1)
        cfg["data"]["root_size"] = 100
    seed = int(cfg["seed"])
    exp = a.exp or (("smoke/" if a.smoke else "") + exp_name(cfg))
    with RunLogger(RESULTS, exp, seed, cfg, use_mlflow=False) as log:
        run(cfg, seed, log, a.device, a.smoke)


def run(cfg, seed, log, device, smoke):
    t_start = time.time()
    rng = set_seed(seed)
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    gen_cpu = torch.Generator()
    gen_cpu.manual_seed(seed)
    N, K, T = cfg["num_clients"], cfg["clients_per_round"], cfg["rounds"]
    dc, ac, tc = cfg["defense"], cfg["attack"], cfg["train"]
    fd = load_fed(cfg["data"]["dataset"], N, cfg["data"].get("partition_clients", 500), cfg["data"]["dirichlet_alpha"], cfg["data"]["val_frac"],
                  cfg["data"]["root_size"], cfg["data"]["val_max"], device, seed, smoke)
    N = len(fd.counts)
    torch.manual_seed(seed)
    model = build_model(fd).to(dev)
    fm = FlatModel(model)
    w = fm.flat()
    trainer = BatchedTrainer(fm, fd, tc["batch"], tc["lr"], amp=bool(tc.get("amp_bf16", True)) and fd.kind == "image")
    net = Network(N, cfg, np.random.default_rng(seed + 1))
    macs = count_macs(model, fd.in_shape)
    cps = 3 * macs / cfg["network"]["macs_per_cycle"]           # forward + backward cycles per sample
    bits_up = 32.0 * fm.d
    net.calibrate_deadline(bits_up, cps, tc["batch"], min(K, N), cfg["network"]["deadline_quantile"],
                           np.random.default_rng(seed + 2))

    attack = ac["type"]
    mal = np.zeros(N, bool)
    if attack != "none":
        mal[rng.choice(N, int(round(ac["ratio"] * N)), replace=False)] = True
    part = np.full(N, 4)
    part[mal] = np.arange(mal.sum()) % 4 if attack == "dba" else 4
    is_bd = attack in BACKDOORS
    mal_t = torch.as_tensor(mal, device=dev)
    part_t = torch.as_tensor(part, device=dev)

    name = dc["name"]
    aegis = name == "aegis"
    defense = Aegis(fm, N, dc, seed, dev, dc["variant"]) if aegis else build_baseline(name, dc, N, fm)
    cloud = Cloud(cfg, q=K / N, rounds=T) if aegis else None
    adv = None
    if attack == "adaptive":
        adv = AdaptiveAttacker(fm.layers, dc["sketch_dim_per_layer"], dc["vae_latent"], 50_000 + seed, dev,
                               ac["adaptive"], dc)
    n_root = len(fd.y_root)
    half = n_root // 2
    root_rng = np.random.default_rng(seed + 3)
    p_mob = cfg["network"]["frac_mobile"]
    from .fl.uav_edge import DEVICE_TYPES
    st_iot, st_mob = DEVICE_TYPES["iot"]["steps"], DEVICE_TYPES["mobile"]["steps"]

    acc_ = dict(tp=0, fp=0, fn=0, tn=0, rej_b=0, n_b=0, rej_b_iot=0, n_b_iot=0, rej_b_mob=0, n_b_mob=0,
                recv=0, recv_mal=0, recv_mal_late=0, recv_late=0, e_client=0.0, e_client_mal=0.0, e_uav=0.0,
                mb_up=0.0, mb_bh=0.0, mb_sketch=0.0, ms_def=0.0, ms_train=0.0, ms_secagg=0.0, stragglers=0,
                sched=0, lam=0.0, secagg_err=0.0, nonfinite=0)
    hist_best = dict(val=-1.0)
    g_prev = None
    diverged = False
    last = {}
    clip = None
    def do_eval(t, n_recv, mal_frac, train_loss, info):
        nonlocal last, hist_best
        v = evaluate(fm, fd, w, fd.x_val, fd.y_val, is_bd)
        te = evaluate(fm, fd, w, fd.x_test, fd.y_test, is_bd)
        row = dict(val_acc=v["acc"], test_acc=te["acc"], test_f1=te["f1"], train_loss=train_loss,
                   n_recv=n_recv, mal_frac_recv=mal_frac, wall=time.time() - t_start, **info)
        if is_bd:
            row.update(val_asr=v["asr"], test_asr=te["asr"])
        log.metric(t, **row)
        last = dict(row, round=t)
        if v["acc"] > hist_best["val"]:
            hist_best = dict(val=v["acc"], test=te["acc"], test_f1=te["f1"], round=t,
                             asr=te.get("asr", float("nan")))
        print(f"[{t:4d}] val {v['acc']:.4f} test {te['acc']:.4f} f1 {te['f1']:.4f} loss {train_loss:.4f}"
              + (f" asr {te['asr']:.3f}" if is_bd else "") + (f" tau {info['tau']:.3g}" if info else ""),
              flush=True)
    for t in range(1, T + 1):
        net.move(rng)
        avail = net.available(rng)
        el = defense.eligible(rng) if hasattr(defense, "eligible") else None
        pool = np.flatnonzero(avail & el) if el is not None else np.flatnonzero(avail)
        if len(pool) < max(2, K // 5):
            pool = np.flatnonzero(avail)
        clients = rng.choice(pool, min(K, len(pool)), replace=False)
        rn = net.round(clients, bits_up, cps, tc["batch"])
        acc_["sched"] += len(clients)
        acc_["stragglers"] += int((~rn.on_time).sum())
        recv = clients[rn.on_time]
        uav = rn.uav[rn.on_time]
        if len(recv) < 2:
            if t % cfg["eval_every"] == 0 or t == T:
                do_eval(t, len(recv), 0.0, float("nan"), {})
            continue
        rt = torch.as_tensor(recv, device=dev)
        m_r = mal[recv]
        poison = None
        if attack in ("label_flip",) + BACKDOORS:
            nt_rows = None
            nt_mask = None
            if attack == "neurotoxin" and g_prev is not None:
                nt_mask = mask_from_global_update(g_prev, ac["neurotoxin_ratio"])
                nt_rows = mal_t[rt]
            poison = Poison(label_flip=mal_t[rt] & (attack == "label_flip"),
                            backdoor=mal_t[rt] & is_bd, part=part_t[rt], nt_mask=nt_mask, nt_rows=nt_rows,
                            frac=ac["backdoor_frac"])
        torch.cuda.synchronize(dev)
        t0 = time.time()
        steps = torch.as_tensor(net.steps[recv], device=dev)
        U, L = trainer.train(w, fd.idx_table[rt], fd.counts[rt], steps, gen, poison=poison)
        torch.cuda.synchronize(dev)
        acc_["ms_train"] += 1e3 * (time.time() - t0)
        train_loss = float(L[torch.as_tensor(~m_r, device=dev)].mean()) if (~m_r).any() else float("nan")
        if attack in MODEL_ATTACKS and m_r.any() and (~m_r).sum() >= 2:
            mi = torch.as_tensor(np.flatnonzero(m_r), device=dev)
            B = U[torch.as_tensor(~m_r, device=dev)]
            if attack == "adaptive":
                U[mi] = adv.craft(B, len(mi), t)
                acc_["lam"] += adv.last_lambda
            else:
                U[mi] = model_poison(attack, U[mi], B, len(recv), ac, gen)

        # a server discards non-finite updates before any aggregation rule (all methods)
        fin = torch.isfinite(U).all(1)
        if not bool(fin.all()):
            keep = fin.cpu().numpy()
            acc_["nonfinite"] += int((~keep).sum())
            U, recv, uav, m_r = U[fin], recv[keep], uav[keep], m_r[keep]
            if len(recv) < 2:                 # every update non-finite: the model has collapsed
                diverged = True
                do_eval(t, len(recv), 0.0, train_loss, {})
                print(f"[{t}] model diverged (all client updates non-finite)", flush=True)
                break
        # ---------------------------------------------------------------- defence
        flagged = None
        info = {}
        if defense.needs_root:
            if aegis:
                n_sh = dc["n_ref_init"] if not defense.ready else dc["n_ref"]
                tab, cnt = root_tables(n_root - half, fd.y_root[half:], n_sh, dc["ref_shard_size"], dc["ref_alpha"],
                                       root_rng, dev)
                tab[:-1] += half
                tab[-1, :] = torch.arange(tab.shape[1], device=dev) % half
                cnt[-1] = half
                st = torch.as_tensor(np.where(root_rng.random(n_sh + 1) < p_mob, st_mob, st_iot), device=dev)
                st[-1] = st_mob
            else:
                tab = torch.arange(n_root, device=dev)[None]
                cnt = torch.as_tensor([n_root], device=dev)
                st = torch.as_tensor([st_mob], device=dev)
            Ur, _ = trainer.train(w, tab, cnt, st, gen, X=fd.x_root, Y=fd.y_root)
        torch.cuda.synchronize(dev)
        t0 = time.time()
        if aegis:
            U_ref, g_c = Ur[:-1], Ur[-1]
            accept, sc, tau, n_sk = defense.step(t, U, recv, U_ref, g_c, truth=m_r, steps=net.steps[recv],
                                                steps_ref=st[:-1].cpu().numpy())
            if clip is None or not math.isfinite(cloud.eps):
                clip = dc["clip_mult"] * float(U_ref.norm(dim=1).median())
            if cloud.sigma > 0 and t == 1:
                clip = dc["clip_mult"] * float(U_ref.norm(dim=1).median())   # fixed C for DP
            ta = time.time()
            acc_t = torch.as_tensor(accept, device=dev)
            agg = cloud.aggregate(U[acc_t], clip, seed * 100_003 + t, cfg["num_uavs"], gen)
            torch.cuda.synchronize(dev)
            acc_["ms_secagg"] += 1e3 * (time.time() - ta)
            acc_["secagg_err"] = max(acc_["secagg_err"], cloud.last_err)
            flagged = ~accept
            acc_["mb_sketch"] += n_sk * defense.sk.dim * 4 / 1e6
            ben = ~m_r
            iot = net.type[recv] == "iot"
            acc_["rej_b"] += int((~accept & ben).sum()); acc_["n_b"] += int(ben.sum())
            acc_["rej_b_iot"] += int((~accept & ben & iot).sum()); acc_["n_b_iot"] += int((ben & iot).sum())
            acc_["rej_b_mob"] += int((~accept & ben & ~iot).sum()); acc_["n_b_mob"] += int((ben & ~iot).sum())
            info = dict(tau=tau, alpha_t=defense.cp.alpha_t)
            fwd = np.bincount(uav[accept], minlength=cfg["num_uavs"]) * bits_up
        else:
            ctx = dict(t=t, clients=recv, g0=Ur[0] if defense.needs_root else None,
                       f=int(round(ac["ratio"] * len(recv))) if attack != "none" else int(round(0.3 * len(recv))),
                       gen=gen, gen_cpu=gen_cpu, w=w)
            agg, fl = defense.aggregate(U, ctx)
            flagged = None if fl is None else fl.cpu().numpy()
            fwd = np.bincount(uav, minlength=cfg["num_uavs"]) * bits_up
        torch.cuda.synchronize(dev)
        acc_["ms_def"] += 1e3 * (time.time() - t0)
        w = w + tc["server_lr"] * agg
        g_prev = agg.detach()
        if not bool(torch.isfinite(w).all()):
            diverged = True

        # ---------------------------------------------------------------- bookkeeping
        if flagged is not None:
            acc_["tp"] += int((flagged & m_r).sum()); acc_["fp"] += int((flagged & ~m_r).sum())
            acc_["fn"] += int((~flagged & m_r).sum()); acc_["tn"] += int((~flagged & ~m_r).sum())
        acc_["recv"] += len(recv); acc_["recv_mal"] += int(m_r.sum())
        if t > T - 100:
            acc_["recv_late"] += len(recv); acc_["recv_mal_late"] += int(m_r.sum())
        e_c = rn.e_cmp + rn.e_com
        acc_["e_client"] += float(e_c.sum()); acc_["e_client_mal"] += float(e_c[mal[clients]].sum())
        e_u = net.uav_energy(rn, fwd)
        acc_["e_uav"] += float(e_u.sum())
        acc_["mb_up"] += len(recv) * bits_up / 8e6
        acc_["mb_bh"] += float(fwd.sum()) / 8e6

        if t % cfg["eval_every"] == 0 or t == T or diverged:
            do_eval(t, len(recv), float(m_r.mean()), train_loss, info)
        if diverged:
            print(f"[{t}] model diverged (non-finite weights)", flush=True)
            break

    R = max(t, 1)
    if aegis and dc.get("dump_scores", False) and defense.dump:
        import pickle
        (log.run_dir / "scores.pkl").write_bytes(pickle.dumps(defense.dump))
    tpr = acc_["tp"] / max(acc_["tp"] + acc_["fn"], 1)
    fpr = acc_["fp"] / max(acc_["fp"] + acc_["tn"], 1)
    det = acc_["tp"] + acc_["fp"] + acc_["fn"] + acc_["tn"] > 0
    log.set_summary(
        dataset=fd.name, defense=name, variant=dc["variant"], attack=attack, ratio=ac["ratio"], seed=seed,
        rounds_run=R, diverged=diverged, d=fm.d, macs=macs, nonfinite_updates=acc_["nonfinite"],
        test_acc=last.get("test_acc"), test_f1=last.get("test_f1"), val_acc=last.get("val_acc"),
        test_asr=last.get("test_asr", float("nan")), best_val_acc=hist_best["val"],
        test_acc_last5=float(np.mean([r["test_acc"] for r in log.history[-5:]])) if log.history else float("nan"),
        test_f1_last5=float(np.mean([r["test_f1"] for r in log.history[-5:]])) if log.history else float("nan"),
        test_asr_last5=float(np.mean([r["test_asr"] for r in log.history[-5:]])) if is_bd and log.history
        else float("nan"),
        test_acc_at_best_val=hist_best.get("test"), best_val_round=hist_best.get("round"),
        tpr=tpr if det else float("nan"), fpr=fpr if det else float("nan"),
        benign_reject=acc_["rej_b"] / max(acc_["n_b"], 1) if aegis else float("nan"),
        benign_reject_iot=acc_["rej_b_iot"] / max(acc_["n_b_iot"], 1) if aegis else float("nan"),
        benign_reject_mobile=acc_["rej_b_mob"] / max(acc_["n_b_mob"], 1) if aegis else float("nan"),
        conformal_alpha=dc["conformal_alpha"],
        mal_frac_received=acc_["recv_mal"] / max(acc_["recv"], 1),
        mal_frac_received_last100=acc_["recv_mal_late"] / max(acc_["recv_late"], 1),
        straggler_frac=acc_["stragglers"] / max(acc_["sched"], 1), deadline_s=net.deadline,
        energy_client_J=acc_["e_client"], energy_client_mal_J=acc_["e_client_mal"],
        energy_uav_kJ=acc_["e_uav"] / 1e3,
        mb_uplink_per_round=acc_["mb_up"] / R, mb_backhaul_per_round=acc_["mb_bh"] / R,
        mb_sketch_per_round=acc_["mb_sketch"] / R,
        ms_train_per_round=acc_["ms_train"] / R, ms_defense_per_round=acc_["ms_def"] / R,
        ms_secagg_per_round=acc_["ms_secagg"] / R, secagg_max_err=acc_["secagg_err"],
        adaptive_lambda_mean=acc_["lam"] / R if attack == "adaptive" else float("nan"),
        dp_epsilon=float(cfg["privacy"]["dp_epsilon"]), dp_sigma=cloud.sigma if cloud else 0.0,
        eps_spent=cloud.eps_spent(R) if cloud else float("inf"),
        lr=tc["lr"], wall_s=time.time() - t_start)
    print({k: v for k, v in log.summary.items() if k in ("test_acc", "test_asr", "tpr", "fpr", "benign_reject",
                                                         "mal_frac_received", "wall_s")}, flush=True)


if __name__ == "__main__":
    main()
