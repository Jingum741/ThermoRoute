"""Schedulers. Baselines are mechanism level reimplementations inside this simulator,
not the original authors' code. obs = dict(Td, Ts, cap, busy)."""
import itertools, numpy as np
from config import K_LEVELS, F_LEVELS, CORES, MPC, T_TRIP, T_SKIN, PREFILL_MULT
from plant import tps, power_active, thermal_step

FULL = max(K_LEVELS)

class Base:
    spec = False
    def __init__(self, soc, Q): self.soc, self.Q = soc, Q
    def act(self, obs): raise NotImplementedError
    def feedback(self, obs_new, a, P_applied): pass

class EdgeMoE(Base):          # full expert budget, thermally unaware, OS governor only
    def act(self, obs): return FULL, 1.0, "performance"

class StaticK4(Base):         # static expert pruning baseline
    def act(self, obs): return 4, 1.0, "performance"

class EdgeLLM(EdgeMoE):       # speculative decoding, thermally unaware
    spec = True

class ReactiveDVFS(Base):     # two threshold hysteresis on frequency
    def __init__(self, soc, Q):
        super().__init__(soc, Q); self.j = 0
    def act(self, obs):
        if obs["Td"] > T_TRIP - 5 or obs["Ts"] > T_SKIN - 0.5:
            self.j = min(self.j + 1, len(F_LEVELS) - 1)
        elif obs["Td"] < T_TRIP - 20 and obs["Ts"] < T_SKIN - 3:
            self.j = max(self.j - 1, 0)
        return FULL, F_LEVELS[self.j], "performance"

def fidrl_state(obs):
    d = int(np.clip((obs["Td"] - 25) // 10, 0, 7))
    s = int(np.clip((obs["Ts"] - 25) // 3, 0, 8))
    c = int(np.argmin(np.abs(np.array(F_LEVELS) - obs["cap"])))
    return d, s, int(obs["busy"]), c

class FiDRL(Base):            # tabular Q learning DVFS agent, trained on SEEDS_TRAIN
    def __init__(self, soc, Q, table=None, eps=0.0, rng=None):
        super().__init__(soc, Q)
        self.table = table if table is not None else np.zeros((8, 9, 2, len(F_LEVELS),
                                                               len(F_LEVELS)))
        self.eps, self.rng = eps, rng or np.random.default_rng(0)
        self.prev = None
    def act(self, obs):
        s = fidrl_state(obs)
        if self.rng.random() < self.eps: a = int(self.rng.integers(len(F_LEVELS)))
        else: a = int(np.argmax(self.table[s]))
        self.prev = (s, a)
        return FULL, F_LEVELS[a], "performance"
    def learn(self, obs_new, reward, lr=0.1, gamma=0.95):
        if self.prev is None: return
        s, a = self.prev; s2 = fidrl_state(obs_new)
        self.table[s + (a,)] += lr * (reward + gamma * self.table[s2].max() - self.table[s + (a,)])

class ThermoRoute(Base):
    """Rolling horizon MPC over (k, f, core).
    Stage cost: quality, speed up to a target, energy per token.
    Constraints (soft): die and skin margins over H steps, plus a lookahead terminal cost that
    projects skin temperature L seconds ahead with the skin time constant, so the controller
    lowers k before the skin limit is reached and restores it as the device cools.
    Duty cycle depends on the action through the closed loop workload profile."""
    def __init__(self, soc, Q, wl_stats, H=None, margin=None, lam=None, nominal=None,
                 T_amb=None, bias=True, w=None, skin_margin=None):
        super().__init__(soc, Q)
        c = dict(MPC); c.update(w or {})
        self.H = int(H or c["H"]); self.L = c["L"]
        self.m = c["margin"] if margin is None else margin
        self.lam = c["lam"] if lam is None else lam
        self.lam_s = c["lam_s"] if lam is None else lam
        self.lam_T = c["lam_T"] if lam is None else lam
        self.ms = c["skin_margin"] if skin_margin is None else skin_margin
        self.mu = c["mu"]
        self.tgt, self.wq, self.ws, self.we = c["tps_target"], c["wq"], c["ws"], c["we"]
        self.p = nominal or soc
        self.T_amb = T_amb                    # None: set from the first skin reading
        self.use_bias = bias
        self.Pbar, self.Rbar, self.Tbar = wl_stats
        self.acts = list(itertools.product(K_LEVELS, F_LEVELS, CORES))
        self.K = np.array([a[0] for a in self.acts], float)
        self.F = np.array([a[1] for a in self.acts], float)
        self.C = np.array([a[2] for a in self.acts])
        self.Qv = np.array([Q[int(k)][0] for k in self.K])
        self.E_ref = float(power_active(self.p, FULL, 1.0, "performance", self.spec)
                           / tps(self.p, FULL, 1.0, "performance", self.spec))
        self.tau_s = self.p["Rsa"] * self.p["Cs"]
        self.bd = self.bs = 0.0
        self.last = None
        self.Tdf = self.Tsf = None

    def _filter(self, obs):
        """First order filter on noisy sensor readings (die is fast, skin is slow)."""
        if self.Tdf is None:
            self.Tdf, self.Tsf = obs["Td"], obs["Ts"]
        else:
            self.Tdf = 0.5 * self.Tdf + 0.5 * obs["Td"]
            self.Tsf = 0.85 * self.Tsf + 0.15 * obs["Ts"]
        return self.Tdf, self.Tsf

    def act(self, obs):
        Tdm, Tsm = self._filter(obs)
        if self.T_amb is None:
            self.T_amb = Tsm                  # device starts in equilibrium with ambient
        cap = obs["cap"]
        f_eff = np.minimum(self.F, cap)
        Pa = power_active(self.p, self.K, f_eff, self.C, self.spec)
        tp = tps(self.p, self.K, f_eff, self.C, self.spec)
        Pi = self.p["idle_w"]
        serve = self.Pbar / (PREFILL_MULT * tp) + self.Rbar / tp
        d = serve / (serve + self.Tbar)                       # action dependent duty
        Td = np.full(len(self.K), Tdm); Ts = np.full(len(self.K), Tsm)
        pen = np.zeros(len(self.K))
        for h in range(self.H):
            a = 1.0 if (h == 0 and obs["busy"]) else d
            Td, Ts = thermal_step(Td, Ts, a * Pa + (1 - a) * Pi, self.p, self.T_amb)
            Td, Ts = Td + self.bd, Ts + self.bs
            pen += self.lam * np.maximum(0, Td - (T_TRIP - self.m)) ** 2
            pen += self.lam_s * np.maximum(0, Ts - (T_SKIN - self.ms)) ** 2
        P_avg = d * Pa + (1 - d) * Pi
        Ts_ss = self.T_amb + P_avg * self.p["Rsa"]
        Ts_L = Ts_ss + (Ts - Ts_ss) * np.exp(-self.L / self.tau_s) + self.bs
        Td_L = Ts_L + Pa * self.p["Rds"]
        pen += self.lam_T * (np.maximum(0, Ts_L - (T_SKIN - self.ms)) ** 2
                             + np.maximum(0, Td_L - (T_TRIP - self.m)) ** 2)
        w = 1.0 if obs["busy"] else d
        util = self.H * w * (self.wq * self.Qv + self.ws * np.minimum(tp, self.tgt) / self.tgt
                             - self.we * (Pa / tp) / self.E_ref)
        sw = np.array([self.mu if (self.last is not None and a != self.last) else 0.0
                       for a in self.acts])
        j = int(np.argmin(-util + pen + sw))
        self.last = self.acts[j]
        self._obs = dict(obs, Td=Tdm, Ts=Tsm)
        return self.acts[j][0], float(min(self.acts[j][1], cap)), self.acts[j][2]

    def feedback(self, obs_new, a, P_applied):
        if not self.use_bias: return
        Tdp, Tsp = thermal_step(self._obs["Td"], self._obs["Ts"], P_applied, self.p, self.T_amb)
        self.bd = float(np.clip(0.9 * self.bd + 0.1 * (obs_new["Td"] - Tdp), -2, 2))
        self.bs = float(np.clip(0.9 * self.bs + 0.1 * (obs_new["Ts"] - Tsp), -0.5, 0.5))

class ThermoRouteSpec(ThermoRoute):
    spec = True

def build(name, soc, Q, wl_stats, plant=None, q_table=None, **kw):
    if name == "EdgeMoE": return EdgeMoE(soc, Q)
    if name == "EdgeLLM": return EdgeLLM(soc, Q)
    if name == "ReactiveDVFS": return ReactiveDVFS(soc, Q)
    if name == "FiDRL": return FiDRL(soc, Q, table=q_table)
    if name == "StaticK4": return StaticK4(soc, Q)
    if name == "ThermoRoute": return ThermoRoute(soc, Q, wl_stats, **kw)
    if name == "ThermoRoute+Spec": return ThermoRouteSpec(soc, Q, wl_stats, **kw)
    if name == "Oracle":      # same MPC with true plant, true ambient, noise free sensing
        return ThermoRoute(soc, Q, wl_stats, nominal=plant.p, T_amb=plant.T_amb, bias=False, **kw)
    raise ValueError(name)
