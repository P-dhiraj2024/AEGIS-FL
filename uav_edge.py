"""Three-tier network: heterogeneous ground devices (IoT sensors, mobiles) -> UAV edge servers -> cloud.

Devices.  Each ground client is an IoT sensor node or a smartphone (DEVICE_TYPES): they differ
in CPU frequency, transmit power, local-step budget, availability and mobility.  IoT nodes are
static; mobiles follow a random walk and re-associate to the nearest UAV every round.

UAVs hover at weighted K-means positions of the clients (core.network.topology) at altitude
h_u.  Uplink rates follow the A2G channel (Eqs. plos, pl, rate; core.network.channel) with
the UAV's bandwidth split equally among its scheduled clients.  Energy: client computation
kappa c D f^2 and communication p S / r (core.network.energy), UAV hovering P(0) (Eq. prop at
V = 0) for the round duration plus the UAV->cloud backhaul.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.network.channel import mean_path_loss_db, rate_bps
from core.network.energy import RotaryWing, hover_power
from core.network.topology import kmeans_placement, nearest_association, sample_clients

# f [GHz] range, p [W], local SGD steps, availability per round, speed [m/s]
DEVICE_TYPES = {
    "iot":    dict(f=(0.5, 1.0), p=0.10, steps=5, avail=0.90, speed=(0.0, 0.0)),
    "mobile": dict(f=(1.5, 2.5), p=0.20, steps=10, avail=0.85, speed=(0.5, 2.0)),
}


@dataclass
class RoundNet:
    uav: np.ndarray          # [K] UAV index of each scheduled client
    t_cmp: np.ndarray        # [K] s
    t_com: np.ndarray        # [K] s
    e_cmp: np.ndarray        # [K] J
    e_com: np.ndarray        # [K] J
    on_time: np.ndarray      # [K] bool (latency <= deadline)


class Network:
    def __init__(self, n_clients: int, cfg: dict, rng: np.random.Generator):
        nc = cfg["network"]
        self.N, self.U = n_clients, cfg["num_uavs"]
        self.area, self.h = nc["area_m"], nc["altitude_m"]
        self.bw, self.fc, self.env = nc["bandwidth_hz"], nc["fc_hz"], nc["env"]
        self.kappa, self.round_s = nc["kappa"], nc["round_dt_s"]
        self.bh_rate, self.bh_p = nc["backhaul_bps"], nc["backhaul_p_w"]
        self.pos = sample_clients(n_clients, self.area, rng, n_hotspots=nc["hotspots"], sigma_m=nc["hotspot_sigma_m"])
        n_mob = int(round(nc["frac_mobile"] * n_clients))
        self.type = np.array(["iot"] * n_clients, dtype=object)
        self.type[rng.permutation(n_clients)[:n_mob]] = "mobile"
        self.is_mobile = self.type == "mobile"
        get = lambda k: np.array([DEVICE_TYPES[t][k] for t in self.type])
        lo, hi = np.array([DEVICE_TYPES[t]["f"] for t in self.type]).T
        self.f = rng.uniform(lo, hi) * 1e9
        self.p = get("p").astype(float)
        self.steps = get("steps").astype(int)
        self.avail = get("avail").astype(float)
        slo, shi = np.array([DEVICE_TYPES[t]["speed"] for t in self.type]).T
        self.speed = rng.uniform(slo, shi)
        self.heading = rng.uniform(0, 2 * np.pi, n_clients)
        self.uav_xy = kmeans_placement(self.pos, self.U, rng)
        self.p_hover = hover_power(RotaryWing())
        self.deadline = None

    def move(self, rng: np.random.Generator):
        """Random-walk mobility of the mobiles for one round (reflecting at the area border)."""
        self.heading += rng.normal(0, 0.5, self.N) * self.is_mobile
        step = (self.speed * self.round_s)[:, None] * np.c_[np.cos(self.heading), np.sin(self.heading)]
        p = self.pos + step
        out = (p < 0) | (p > self.area)
        self.heading[out.any(1)] += np.pi
        self.pos = np.clip(p, 0, self.area)

    def available(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(self.N) < self.avail

    def round(self, clients: np.ndarray, bits_up: float, cycles_per_sample: float, batch: int) -> RoundNet:
        uav = nearest_association(self.uav_xy, self.pos[clients])
        d = np.linalg.norm(self.pos[clients] - self.uav_xy[uav], axis=1)
        share = np.bincount(uav, minlength=self.U)[uav]
        bw = self.bw / np.maximum(share, 1)
        pl = mean_path_loss_db(self.h, d, self.fc, self.env)
        r = rate_bps(self.p[clients], pl, bw)
        cyc = cycles_per_sample * batch * self.steps[clients]
        t_cmp = cyc / self.f[clients]
        t_com = bits_up / r
        e_cmp = self.kappa * cyc * self.f[clients] ** 2
        e_com = self.p[clients] * t_com
        lat = t_cmp + t_com
        on = np.ones(len(clients), bool) if self.deadline is None else lat <= self.deadline
        return RoundNet(uav, t_cmp, t_com, e_cmp, e_com, on)

    def calibrate_deadline(self, bits_up: float, cycles_per_sample: float, batch: int, k: int,
                           quantile: float, rng: np.random.Generator, trials: int = 50):
        """Deadline = `quantile` of client latency over random schedules of k clients (fixed per run)."""
        lat = []
        for _ in range(trials):
            c = rng.choice(self.N, k, replace=False)
            rn = self.round(c, bits_up, cycles_per_sample, batch)
            lat.append(rn.t_cmp + rn.t_com)
        self.deadline = float(np.quantile(np.concatenate(lat), quantile))

    def uav_energy(self, rn: RoundNet, fwd_bits_per_uav: np.ndarray, extra_s: float = 0.0) -> np.ndarray:
        """Hover energy for the round duration (slowest on-time client) + backhaul energy, per UAV [J]."""
        lat = np.where(rn.on_time, rn.t_cmp + rn.t_com, 0.0)
        dur = np.zeros(self.U)
        np.maximum.at(dur, rn.uav, lat)
        if self.deadline is not None:
            dur = np.minimum(np.maximum(dur, 0.0), self.deadline)
        t_bh = fwd_bits_per_uav / self.bh_rate
        return self.p_hover * (dur + extra_s + t_bh) + self.bh_p * t_bh
