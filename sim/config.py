"""ThermoRoute v3: single source of truth for every constant.
Tags: [measured] produced by Step 1 / Step 2 of the notebook, [spec] vendor specification (verify),
[repr] representative value declared in the paper, [assumed] modelling assumption."""
import os

ROOT = os.environ.get("TR_ROOT", os.getcwd())
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
TRACE_DIR = os.path.join(RESULTS_DIR, "traces")
FIG_DIR = os.environ.get("TR_FIG_DIR",
                         os.path.join(os.path.expanduser("~"), "Downloads", "ThermoRoute"))
for _d in (DATA_DIR, RESULTS_DIR, TRACE_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# Experiment grid
SEEDS_TEST = list(range(20))            # 20 seeds: Holm corrected Wilcoxon can reach p < 0.05
SEEDS_TRAIN = list(range(100, 120))           # FiDRL training only, disjoint from test seeds
SIM_SECONDS = 1800
AMBIENTS = [25.0, 30.0, 35.0]
WORKLOADS = {"chat": (3.0, 12.0), "agent": (0.5, 2.0)}   # think time range in s [assumed]
MAIN_COND = dict(soc="SoC-A", amb=25.0, wl="agent")      # main table
STRESS_COND = dict(soc="SoC-A", amb=35.0, wl="agent")    # sweeps run here (constraints active)
N_JOBS = max(1, (os.cpu_count() or 2))
if os.environ.get("THERMOROUTE_QUICK"):                  # smoke test only, never for the paper
    SEEDS_TEST, SEEDS_TRAIN, SIM_SECONDS = [0, 1], [100], 300

# Model (OLMoE 1B 7B: 64 experts, top 8)
HF_MODEL = "allenai/OLMoE-1B-7B-0924"
K_LEVELS = [2, 4, 6, 8]
BPP = 0.56                     # bytes per parameter, Q4_K_M style on device weights [assumed]

# DVFS and throughput
F_LEVELS = [1.0, 0.85, 0.70, 0.55, 0.40, 0.30]
CORES = ["performance", "efficiency"]
TPS_FREQ_EXP = 0.7             # memory bound decode, achieved bandwidth ~ f^0.7 [assumed]
PREFILL_MULT = 8.0             # prefill tokens/s relative to decode tokens/s [assumed]
SLO_TPS = 20.0                 # per response decode speed SLO, tok/s [assumed]
BATTERY_J = 19.25 * 3600.0     # 5000 mAh at 3.85 V

# SoC profiles. Power = idle + compute_w[core] f^3 + e_byte x bytes/s moved.
SOCS = {
    "SoC-A": dict(label="Flagship class", mem_bw_gbs=67.2, bw_eff=0.35,       # mem_bw [spec]
                  idle_w=0.5, e_byte=8.0e-11,
                  compute_w={"performance": 4.5, "efficiency": 1.6},
                  core_eff={"performance": 1.0, "efficiency": 0.55},
                  Cd=4.5, Rds=4.5, Cs=85.0, Rsa=3.6),
    "SoC-B": dict(label="Mid range class", mem_bw_gbs=51.2, bw_eff=0.35,
                  idle_w=0.4, e_byte=8.0e-11,
                  compute_w={"performance": 3.6, "efficiency": 1.3},
                  core_eff={"performance": 1.0, "efficiency": 0.55},
                  Cd=4.0, Rds=5.0, Cs=75.0, Rsa=3.9),
}

# OS thermal governor, identical for every method [repr]. Progressive step governor:
# every GOV_PERIOD s the cap moves one F_LEVELS step down while above the trip point and
# one step up once below the release point. Die emergency cap at T_CRIT.
GOV_PERIOD = 2
T_TRIP, T_RELEASE, T_CRIT = 85.0, 80.0, 95.0
T_SKIN, T_SKIN_RELEASE = 43.0, 42.0

# Controller side uncertainty
SENSOR_NOISE = 0.5             # C, std
MISMATCH_DEFAULT = 0.10        # +/- uniform fraction on R and C of the true plant

# ThermoRoute MPC defaults
MPC = dict(H=10, L=180, margin=3.0, skin_margin=1.0, lam=5.0, lam_s=5.0, lam_T=5.0,
           mu=1.0, tps_target=30.0, wq=10.0, ws=6.0, we=2.0)
MPC_WEIGHTS_JSON = os.path.join(RESULTS_DIR, "mpc_weights.json")   # written by the tuning step
if os.path.exists(MPC_WEIGHTS_JSON):
    import json as _json
    MPC.update(_json.load(open(MPC_WEIGHTS_JSON)))
TUNE_GRID = dict(wq=[10.0, 20.0, 40.0], we=[0.5, 1.0, 2.0], ws=[6.0, 12.0])
TUNE_CONDS = [("SoC-A", 25.0, "agent"), ("SoC-A", 35.0, "agent"), ("SoC-A", 35.0, "chat")]

# Speculative decoding (EdgeLLM style) [assumed]; expert union per verify step is measured
SPEC = dict(alpha=0.7, gamma=4, draft_cost=0.10, extra_w=0.8)

# Sweeps (run on STRESS_COND)
SWEEP_H = [5, 10, 20, 40]
SWEEP_M = [0.5, 1.0, 2.0]      # skin margin (the skin limit binds; the die limit does not)
SWEEP_LAM = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
SWEEP_MISMATCH = [0.0, 0.1, 0.2, 0.3]

METHODS = ["EdgeMoE", "EdgeLLM", "ReactiveDVFS", "FiDRL", "StaticK4",
           "ThermoRoute", "ThermoRoute+Spec", "Oracle"]
