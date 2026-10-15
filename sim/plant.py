"""Two node (die, skin) RC thermal plant, memory bound throughput and power models,
and the progressive OS thermal governor shared by every method."""
import numpy as np
from config import (TPS_FREQ_EXP, BPP, F_LEVELS, GOV_PERIOD, T_TRIP, T_RELEASE, T_CRIT,
                    T_SKIN, T_SKIN_RELEASE, SENSOR_NOISE, SPEC)
from model_profile import param_counts, load_union

SHARED_P, EXPERT_P, N_EXPERTS = param_counts()
UNION = load_union()
ACC = (1 - SPEC["alpha"] ** (SPEC["gamma"] + 1)) / (1 - SPEC["alpha"])   # expected tokens/step

def _core(p, key, core):
    return np.vectorize(lambda c: p[key][c])(core) if np.ndim(core) else p[key][core]

def bandwidth(p, f, core):
    """Bytes/s moved while decoding (memory bound)."""
    return p["mem_bw_gbs"] * 1e9 * p["bw_eff"] * np.asarray(f) ** TPS_FREQ_EXP \
        * _core(p, "core_eff", core)

def bytes_per_token(k, spec=False):
    """Plain decode: shared + k expert slots. Speculative decode: gamma draft passes plus one
    verify pass that reads the union of experts selected by the gamma+1 verified tokens."""
    k = np.asarray(k, float)
    plain = BPP * (SHARED_P + k * EXPERT_P)
    if not spec:
        return plain
    u = np.vectorize(lambda kk: UNION[int(kk)])(k) if np.ndim(k) else UNION[int(k)]
    step = SPEC["gamma"] * SPEC["draft_cost"] * plain + BPP * (SHARED_P + u * EXPERT_P)
    return step / ACC

def tps(p, k, f, core, spec=False):
    return bandwidth(p, f, core) / bytes_per_token(k, spec)

def power_active(p, k, f, core, spec=False):
    """At a given frequency the memory system is saturated, so k changes tokens per byte,
    not watts. Fewer experts finish a response sooner and lengthen idle time."""
    f = np.asarray(f)
    comp = (_core(p, "compute_w", core) + (SPEC["extra_w"] if spec else 0.0)) * f ** 3
    return p["idle_w"] + comp + p["e_byte"] * bandwidth(p, f, core) + 0 * np.asarray(k, float)

def thermal_step(Td, Ts, P, p, T_amb, dt=1.0):
    Td_n = Td + dt / p["Cd"] * (P - (Td - Ts) / p["Rds"])
    Ts_n = Ts + dt / p["Cs"] * ((Td - Ts) / p["Rds"] - (Ts - T_amb) / p["Rsa"])
    return Td_n, Ts_n

class Plant:
    def __init__(self, soc, T_amb, rng, mismatch):
        self.p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in soc.items()}
        for key in ("Cd", "Rds", "Cs", "Rsa"):
            self.p[key] *= 1 + rng.uniform(-mismatch, mismatch)
        self.T_amb, self.rng = T_amb, rng
        self.Td = self.Ts = T_amb
        self.jd = self.js = 0                 # governor step index into F_LEVELS
        self.tick = 0

    @property
    def gov_die(self): return self.jd > 0
    @property
    def gov_skin(self): return self.js > 0

    def cap(self):
        return min(F_LEVELS[self.jd], F_LEVELS[self.js])

    def step(self, P):
        self.Td, self.Ts = thermal_step(self.Td, self.Ts, P, self.p, self.T_amb)
        self.tick += 1
        last = len(F_LEVELS) - 1
        if self.Td >= T_CRIT:
            self.jd = last
        elif self.tick % GOV_PERIOD == 0:
            if self.Td >= T_TRIP: self.jd = min(self.jd + 1, last)
            elif self.Td <= T_RELEASE: self.jd = max(self.jd - 1, 0)
        if self.tick % GOV_PERIOD == 0:
            if self.Ts >= T_SKIN: self.js = min(self.js + 1, last)
            elif self.Ts <= T_SKIN_RELEASE: self.js = max(self.js - 1, 0)

    def measure(self, noise=True):
        n = self.rng.normal(0, SENSOR_NOISE, 2) if noise else np.zeros(2)
        return self.Td + n[0], self.Ts + n[1]
