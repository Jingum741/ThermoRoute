"""Episode simulation, FiDRL training, and the full experiment grid with checkpointing
(a Colab disconnect resumes where it stopped)."""
import os, time, itertools, pickle, hashlib, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from config import *
from model_profile import load_quality
from plant import Plant, tps, power_active
from workload import Workload, workload_stats
from controllers import build, FiDRL

Q = load_quality()
LOGPPL = {k: np.log(v[1]) for k, v in Q.items()}
SAMPLE_DIR = os.path.join(RESULTS_DIR, "samples"); os.makedirs(SAMPLE_DIR, exist_ok=True)
RUNS_CSV = os.path.join(RESULTS_DIR, "all_runs.csv")
FIDRL_PKL = os.path.join(RESULTS_DIR, "fidrl_tables.pkl")

def episode(method, soc_key, amb, wl, seed, mismatch=MISMATCH_DEFAULT, kw=None,
            q_table=None, trace=False, agent=None):
    rng = np.random.default_rng(seed * 7919 + int(amb) * 31 + (0 if wl == "chat" else 1))
    soc = SOCS[soc_key]
    plant = Plant(soc, amb, np.random.default_rng(rng.integers(1 << 31)), mismatch)
    work = Workload(WORKLOADS[wl], np.random.default_rng(rng.integers(1 << 31)))
    ctrl = agent or build(method, soc, Q, workload_stats(WORKLOADS[wl]), plant=plant,
                          q_table=q_table, **(kw or {}))
    noisy = method != "Oracle"
    Td_m, Ts_m = plant.measure(noisy)
    rec = {x: [] for x in ("Td", "Ts", "k", "f", "tps", "a", "P", "cap")}
    E = tok = qtok = lptok = die_s = skin_s = anyg = 0.0
    k_time = {k: 0.0 for k in K_LEVELS}
    ovh, pk_d, pk_s = [], plant.Td, plant.Ts
    for t in range(SIM_SECONDS):
        busy = work.busy()
        obs = dict(Td=Td_m, Ts=Ts_m, cap=plant.cap(), busy=busy)
        t0 = time.perf_counter(); k, f, core = ctrl.act(obs); dt_ctrl = time.perf_counter() - t0
        if busy: ovh.append(dt_ctrl)                    # overhead on active decisions only
        fe = min(f, plant.cap())
        r = float(tps(soc, k, fe, core, ctrl.spec))
        a, dec = work.advance(r)
        P = a * float(power_active(soc, k, fe, core, ctrl.spec)) + (1 - a) * soc["idle_w"]
        die_s += plant.gov_die; skin_s += plant.gov_skin; anyg += (plant.gov_die or plant.gov_skin)
        cap_now = plant.cap()
        plant.step(P); E += P
        pk_d, pk_s = max(pk_d, plant.Td), max(pk_s, plant.Ts)
        tok += dec; qtok += dec * Q[k][0]; lptok += dec * LOGPPL[k]
        k_time[k] += a
        Td_m, Ts_m = plant.measure(noisy)
        obs_new = dict(Td=Td_m, Ts=Ts_m, cap=plant.cap(), busy=work.busy())
        ctrl.feedback(obs_new, a, P)
        if isinstance(ctrl, FiDRL) and ctrl.eps > 0:
            ctrl.learn(obs_new, dec / 10 - 5.0 * plant.gov_die
                       - 2.0 * max(0.0, plant.Ts - T_SKIN) - 0.1 * P)
        if trace:
            for x, v in zip(rec, (plant.Td, plant.Ts, k, fe, r * (a > 0), a, P, cap_now)):
                rec[x].append(v)
    lat = np.array([d[0] for d in work.done]); ttft = np.array([d[1] for d in work.done])
    dtps = np.array([d[2] for d in work.done])
    act = sum(k_time.values()) or 1.0
    nan = float("nan")
    res = dict(method=method, soc=soc_key, amb=amb, wl=wl, seed=seed, mismatch=mismatch,
               tokens=tok, qwt=qtok,
               eff_ppl=float(np.exp(lptok / tok)) if tok > 0 else nan,   # token weighted geometric
               peak_die=pk_d, peak_skin=pk_s,
               die_trip_s=die_s, skin_cap_s=skin_s, throttle_s=anyg, energy_J=E,
               j_per_tok=E / tok if tok else nan,
               tok_per_charge=tok / E * BATTERY_J if E else nan,
               responses=len(lat),
               lat_p50=np.percentile(lat, 50) if len(lat) else nan,
               lat_p95=np.percentile(lat, 95) if len(lat) else nan,
               ttft_p95=np.percentile(ttft, 95) if len(lat) else nan,
               slo_viol_pct=100 * np.mean(dtps < SLO_TPS) if len(lat) else nan,
               ovh_med_ms=1e3 * np.median(ovh) if ovh else nan,
               ovh_p99_ms=1e3 * np.percentile(ovh, 99) if ovh else nan,
               **{f"frac_k{k}": k_time[k] / act for k in K_LEVELS})
    for c in ("H", "skin_margin", "lam"):
        res[c] = (kw or {}).get(c, nan)
    return res, (rec if trace else None), dict(lat=lat, ovh=np.array(ovh) * 1e3)

def train_fidrl(soc_key, wl, schedule=(0.3, 0.15, 0.05)):
    """Episodes on training seeds at every ambient; exploration decays per pass.
    Plant, workload and controller state are rebuilt every episode."""
    agent = FiDRL(SOCS[soc_key], Q, eps=schedule[0], rng=np.random.default_rng(1))
    for eps in schedule:
        agent.eps = eps
        for s in SEEDS_TRAIN:
            for amb in AMBIENTS:
                agent.prev = None
                episode("FiDRL", soc_key, amb, wl, s, agent=agent)
    return agent.table

def get_fidrl_tables():
    if os.path.exists(FIDRL_PKL):
        with open(FIDRL_PKL, "rb") as f:
            return pickle.load(f)
    tables = {}
    for soc_key in SOCS:
        for wl in WORKLOADS:
            tables[(soc_key, wl)] = train_fidrl(soc_key, wl)
            print("FiDRL trained", soc_key, wl, flush=True)
    with open(FIDRL_PKL, "wb") as f:
        pickle.dump(tables, f)
    return tables

def _tune_one(args):
    w, (sk, amb, wl), s = args
    r, _, _ = episode("ThermoRoute", sk, amb, wl, s, kw=dict(w=w))
    return dict(**w, soc=sk, amb=amb, wl=wl, seed=s, qwt_per_J=r["qwt"] / r["energy_J"],
                throttle_s=r["throttle_s"], slo=r["slo_viol_pct"], lat_p95=r["lat_p95"])

def tune_weights(n_seeds=2):
    """Select MPC stage weights on TRAINING seeds only (never on test seeds).
    Rule: among settings with mean throttle <= 1 % of the episode and mean SLO violation
    <= 5 %, take the highest quality weighted tokens per joule."""
    import json
    if os.path.exists(MPC_WEIGHTS_JSON):
        best = json.load(open(MPC_WEIGHTS_JSON)); MPC.update(best)
        print("MPC weights (saved):", best); return best
    grid = [dict(zip(TUNE_GRID, v)) for v in itertools.product(*TUNE_GRID.values())]
    jobs = list(itertools.product(grid, TUNE_CONDS, SEEDS_TRAIN[:n_seeds]))
    with ProcessPoolExecutor(N_JOBS) as ex:
        df = pd.DataFrame(list(ex.map(_tune_one, jobs, chunksize=2)))
    df.to_csv(os.path.join(RESULTS_DIR, "mpc_tuning.csv"), index=False)
    g = df.groupby(list(TUNE_GRID))[["qwt_per_J", "throttle_s", "slo"]].mean().reset_index()
    ok = g[(g.throttle_s <= 0.01 * SIM_SECONDS) & (g.slo <= 5.0)]
    pick = (ok if len(ok) else g.sort_values("throttle_s").head(3)).sort_values(
        "qwt_per_J", ascending=False).iloc[0]
    best = {k: float(pick[k]) for k in TUNE_GRID}
    json.dump(best, open(MPC_WEIGHTS_JSON, "w")); MPC.update(best)
    print(g.round(3).to_string(index=False)); print("selected MPC weights:", best)
    return best

def job_list(tables):
    jobs = []
    def add(tag, m, sk, amb, wl, s, mismatch=MISMATCH_DEFAULT, kw=None, save=False):
        jid = f"{tag}|{m}|{sk}|{amb:g}|{wl}|{s}|{mismatch:g}|" + \
              ",".join(f"{a}={b:g}" for a, b in sorted((kw or {}).items()))
        jobs.append(dict(jid=jid, tag=tag, save=save, method=m, soc_key=sk, amb=amb, wl=wl,
                         seed=s, mismatch=mismatch, kw=kw,
                         q_table=tables[(sk, wl)] if m == "FiDRL" else None))
    for m, sk, amb, wl, s in itertools.product(METHODS, SOCS, AMBIENTS, WORKLOADS, SEEDS_TEST):
        add("main", m, sk, amb, wl, s, save=True)
    sc = STRESS_COND
    for H, mg, s in itertools.product(SWEEP_H, SWEEP_M, SEEDS_TEST):
        add("HM", "ThermoRoute", sc["soc"], sc["amb"], sc["wl"], s, kw=dict(H=H, skin_margin=mg))
    for lam, s in itertools.product(SWEEP_LAM, SEEDS_TEST):
        add("lam", "ThermoRoute", sc["soc"], sc["amb"], sc["wl"], s, kw=dict(lam=lam))
    for mm, m, s in itertools.product(SWEEP_MISMATCH, ["EdgeMoE", "ReactiveDVFS", "FiDRL",
                                                     "ThermoRoute"], SEEDS_TEST):
        add("mismatch", m, sc["soc"], sc["amb"], sc["wl"], s, mismatch=mm)
    return jobs

def run_job(job):
    job = dict(job)
    jid, tag, save = job.pop("jid"), job.pop("tag"), job.pop("save")
    res, rec, extra = episode(**job, trace=save)
    res["sweep"], res["jid"] = tag, jid
    tr = None
    if rec is not None:
        if job["seed"] == 0:
            tr = rec
        np.savez_compressed(os.path.join(SAMPLE_DIR, hashlib.md5(jid.encode()).hexdigest()[:16] + ".npz"),
                            jid=jid, lat=extra["lat"], ovh=extra["ovh"])
    return res, tr

def run_all(flush_every=40):
    tune_weights()
    tables = get_fidrl_tables()
    jobs = job_list(tables)
    done = set(pd.read_csv(RUNS_CSV, usecols=["jid"]).jid) if os.path.exists(RUNS_CSV) else set()
    todo = [j for j in jobs if j["jid"] not in done]
    print(f"jobs total {len(jobs)}, done {len(done)}, to run {len(todo)}, workers {N_JOBS}",
          flush=True)
    buf, t0 = [], time.time()
    def flush():
        if not buf: return
        df = pd.DataFrame(buf)
        if os.path.exists(RUNS_CSV):
            cols = pd.read_csv(RUNS_CSV, nrows=0).columns
            df.reindex(columns=cols).to_csv(RUNS_CSV, mode="a", header=False, index=False)
        else:
            df.to_csv(RUNS_CSV, index=False)
        buf.clear()
    with ProcessPoolExecutor(N_JOBS) as ex:
        for i, (res, rec) in enumerate(ex.map(run_job, todo, chunksize=2), 1):
            buf.append(res)
            if rec is not None:
                name = f"{res['method']}_{res['soc']}_{int(res['amb'])}_{res['wl']}.npz"
                np.savez_compressed(os.path.join(TRACE_DIR, name.replace("+", "p")),
                                    **{k: np.array(v) for k, v in rec.items()})
            if len(buf) >= flush_every:
                flush()
                print(f"{i}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    flush()
    print("all runs saved to", RUNS_CSV)
