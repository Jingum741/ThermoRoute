"""Per condition mean and 95% t CI, paired Wilcoxon (ThermoRoute vs each method) with
direction and Holm correction, and IEEE LaTeX tables for MAIN_COND and STRESS_COND."""
import os, numpy as np, pandas as pd
from scipy import stats
from config import RESULTS_DIR, MAIN_COND, STRESS_COND, METHODS

METRICS = ["peak_die", "peak_skin", "throttle_s", "eff_ppl", "lat_p95", "ttft_p95",
           "slo_viol_pct", "j_per_tok", "tok_per_charge", "responses", "ovh_med_ms"]
HIGHER_BETTER = {"tok_per_charge", "responses"}
TEST = ["throttle_s", "peak_skin", "eff_ppl", "lat_p95", "slo_viol_pct", "j_per_tok",
        "tok_per_charge", "responses"]
TAB_COLS = [("peak_skin", r"Peak skin ($^\circ$C)", 2), ("throttle_s", "Throttle (s)", 1),
            ("eff_ppl", "PPL", 2), ("lat_p95", "p95 lat. (s)", 2), ("slo_viol_pct", "SLO viol. (\\%)", 1),
            ("j_per_tok", "J/token", 4), ("tok_per_charge", "Tokens/charge ($10^3$)", 1)]

def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

def cond(d, c):
    return d[(d.soc == c["soc"]) & (d.amb == c["amb"]) & (d.wl == c["wl"])]

def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]; m = len(order); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); out[i] = run
    return out

def wilcoxon_table(dc):
    rows = []
    tr = dc[dc.method == "ThermoRoute"].set_index("seed")
    for m in METHODS:
        if m == "ThermoRoute": continue
        o = dc[dc.method == m].set_index("seed")
        for k in TEST:
            a, b = tr[k].align(o[k], join="inner")
            diff = (a - b).to_numpy(); p = np.nan
            if len(diff) > 1 and np.any(diff != 0):
                p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
            md = float(np.median(diff))
            better = (md > 0) if k in HIGHER_BETTER else (md < 0)
            rows.append({"baseline": m, "metric": k, "thermoroute_mean": a.mean(),
                         "baseline_mean": b.mean(), "median_diff": md, "p": p,
                         "direction": "tie" if md == 0 else
                         ("ThermoRoute better" if better else "ThermoRoute worse")})
    w = pd.DataFrame(rows); w["p_holm"] = holm(w.p.to_numpy())
    w["significant_0.05"] = w.p_holm < 0.05
    return w

def latex_table(dc, caption, label):
    lines = [r"\begin{table}[!t]", r"\centering", r"\caption{" + caption + "}",
             r"\label{" + label + "}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(TAB_COLS) + "}", r"\hline",
             "Method & " + " & ".join(h for _, h, _ in TAB_COLS) + r" \\", r"\hline"]
    for m in METHODS:
        r = dc[dc.method == m]; cells = []
        for k, _, dp in TAB_COLS:
            sc = 1e-3 if k == "tok_per_charge" else 1.0
            cells.append(f"${r[k].mean() * sc:.{dp}f}\\pm{ci95(r[k]) * sc:.{dp}f}$")
        name = r"\textbf{" + m + "}" if m == "ThermoRoute" else m
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)

def main():
    d = pd.read_csv(os.path.join(RESULTS_DIR, "all_runs.csv"))
    mr = d[d.sweep == "main"]
    g = mr.groupby(["soc", "amb", "wl", "method"])
    summ = g[METRICS].mean().add_suffix("_mean").join(g[METRICS].agg(ci95).add_suffix("_ci"))
    summ.reset_index().to_csv(os.path.join(RESULTS_DIR, "summary_all_conditions.csv"), index=False)
    for tag, c in (("main", MAIN_COND), ("stress", STRESS_COND)):
        dc = cond(mr, c)
        wilcoxon_table(dc).to_csv(os.path.join(RESULTS_DIR, f"wilcoxon_{tag}.csv"), index=False)
        cap = (f"Results at {c['soc']}, {c['amb']:g} \\textdegree C ambient, {c['wl']} workload "
               f"(mean $\\pm$ 95\\% CI over {dc.seed.nunique()} seeds)")
        with open(os.path.join(RESULTS_DIR, f"table_{tag}.tex"), "w") as f:
            f.write(latex_table(dc, cap, f"tab:{tag}"))
    print(open(os.path.join(RESULTS_DIR, "table_main.tex")).read())

if __name__ == "__main__":
    main()
