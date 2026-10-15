"""All result figures (IEEE style: Times New Roman, 8 pt, 800 dpi, single 3.5 in / double
7.16 in). Every number is read from results/ so figures always agree with the tables.
Panel tags sit below the x label; one shared legend sits under each figure."""
import os, glob
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from config import (RESULTS_DIR, TRACE_DIR, DATA_DIR, FIG_DIR, MAIN_COND, STRESS_COND, METHODS,
                    K_LEVELS, T_TRIP, T_SKIN, AMBIENTS, SOCS, SWEEP_H, SWEEP_M)

for _f in fm.findSystemFonts():
    if any(s in os.path.basename(_f).lower() for s in ("times", "liberationserif")):
        try: fm.fontManager.addfont(_f)
        except Exception: pass

SINGLE, DOUBLE, DPI = 3.5, 7.16, 800
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False, "axes.grid": True,
    "grid.linewidth": 0.3, "grid.alpha": 0.4, "savefig.dpi": DPI,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42})

STYLE = {  # color, marker, linestyle, hatch: readable in grayscale print
    "EdgeMoE":          ("#1f77b4", "o", "-", ""),
    "EdgeLLM":          ("#ff7f0e", "s", "--", "//"),
    "ReactiveDVFS":     ("#8c564b", "v", "-.", "\\\\"),
    "FiDRL":            ("#2ca02c", "^", ":", "xx"),
    "StaticK4":         ("#9467bd", "D", (0, (3, 1, 1, 1)), ".."),
    "ThermoRoute":      ("#d62728", "*", "-", ""),
    "ThermoRoute+Spec": ("#e377c2", "P", "--", "--"),
    "Oracle":           ("#7f7f7f", "x", ":", "++"),
}
LABEL = {"peak_skin": "Peak skin temp. (°C)", "throttle_s": "Throttle time (s)",
         "eff_ppl": "Effective perplexity", "j_per_tok": "Energy per token (J)",
         "lat_p95": "p95 latency (s)", "tok_per_charge": "Tokens per charge"}

def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"))
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    plt.close(fig); print("saved", name)

def legend_below(fig, handles, labels, ncol=None):
    fig.legend(handles, labels, ncol=ncol or min(len(labels), 4), loc="upper center",
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.0)

def tag(ax, letter, xlabel):
    ax.set_xlabel(f"{xlabel}\n({letter})")

def runs(sweep="main"):
    d = pd.read_csv(os.path.join(RESULTS_DIR, "all_runs.csv"))
    return d[d.sweep == sweep]

def cond(d, c):
    return d[(d.soc == c["soc"]) & (d.amb == c["amb"]) & (d.wl == c["wl"])]

def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

def trace(method, c):
    name = f"{method}_{c['soc']}_{int(c['amb'])}_{c['wl']}.npz".replace("+", "p")
    return np.load(os.path.join(TRACE_DIR, name))

def cname(c):
    return f"{c['soc']}, {c['amb']:g} °C, {c['wl']}"

# ---------------------------------------------------------------------------------------------
def fig_A():
    """Measured quality, analytic phone speed, measured expert union (double column)."""
    from plant import tps
    kp = pd.read_csv(os.path.join(DATA_DIR, "k_profile.csv"))
    un = pd.read_csv(os.path.join(DATA_DIR, "expert_union.csv"))
    p = SOCS[MAIN_COND["soc"]]
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 1.75))
    ax[0].errorbar(kp.k, kp.ppl, yerr=kp.get("ppl_ci", 0 * kp.ppl), fmt="o-", color="#d62728",
                   capsize=2, elinewidth=0.6)
    ax[0].set_ylabel("WikiText 103 perplexity"); ax[0].set_xticks(K_LEVELS)
    tag(ax[0], "a", "Active experts $k$")
    h1, = ax[1].plot(K_LEVELS, [float(tps(p, k, 1.0, "performance")) for k in K_LEVELS], "o-",
                     color="#1f77b4", label="Plain decoding")
    h2, = ax[1].plot(K_LEVELS, [float(tps(p, k, 1.0, "performance", True)) for k in K_LEVELS],
                     "s--", color="#ff7f0e", label="Speculative decoding")
    ax[1].set_ylabel("Phone decode speed (tok/s)"); ax[1].set_xticks(K_LEVELS)
    tag(ax[1], "b", "Active experts $k$")
    for i, k in enumerate(K_LEVELS):
        u = un[un.k == k].sort_values("window")
        ax[2].plot(u.window, u.union, marker="os^D"[i], ls="-", color=plt.cm.Greys(0.35 + 0.18 * i),
                   label=f"$k={k}$")
    ax[2].set_ylabel("Distinct experts per layer"); ax[2].set_xticks(sorted(un.window.unique()))
    tag(ax[2], "c", "Consecutive tokens in window")
    fig.tight_layout(w_pad=1.0)
    h3, l3 = ax[2].get_legend_handles_labels()
    legend_below(fig, [h1, h2] + h3, [h1.get_label(), h2.get_label()] + l3, ncol=6)
    save(fig, "fig_A_k_profile")

def fig_B():
    """Temporal dynamics, STRESS_COND, seed 0 (double column, stacked panels)."""
    c = STRESS_COND
    show = ["EdgeMoE", "ReactiveDVFS", "FiDRL", "StaticK4", "ThermoRoute"]
    fig, ax = plt.subplots(5, 1, figsize=(DOUBLE, 5.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1, 0.75, 1, 0.75]))
    for m in show:
        tr = trace(m, c); t = np.arange(len(tr["Td"])); col, _, ls, _ = STYLE[m]
        ax[0].plot(t, tr["Td"], color=col, ls=ls, lw=0.9, label=m)
        ax[1].plot(t, tr["Ts"], color=col, ls=ls, lw=0.9)
        if m == "ThermoRoute":                            # k only matters while decoding
            kk = pd.Series(np.where(tr["a"] > 0, tr["k"], np.nan)).rolling(
                30, center=True, min_periods=1).mean()
            ax[2].plot(t, kk, color=col, ls=ls, lw=0.9)
        ax[3].plot(t, pd.Series(tr["tps"]).rolling(15, center=True, min_periods=1).mean(),
                   color=col, ls=ls, lw=0.9)
        ax[4].step(t, tr["cap"], color=col, ls=ls, lw=0.8, where="post")
    ax[0].axhline(T_TRIP, color="k", ls="--", lw=0.6)
    ax[1].axhline(T_SKIN, color="k", ls="--", lw=0.6)
    ax[0].set_ylabel("Die temp. (°C)"); ax[1].set_ylabel("Skin temp. (°C)")
    ax[2].set_ylabel("ThermoRoute $k$\n(30 s mean)"); ax[2].set_yticks(K_LEVELS)
    ax[2].set_ylim(min(K_LEVELS) - 0.5, max(K_LEVELS) + 0.5)
    ax[3].set_ylabel("Tok/s (15 s avg.)"); ax[4].set_ylabel("Governor cap")
    ax[4].set_xlabel("Time (s)")
    for a, l in zip(ax, "abcde"):
        a.text(1.005, 0.5, f"({l})", transform=a.transAxes, va="center", ha="left")
    fig.align_ylabels(ax); fig.tight_layout(h_pad=0.3)
    h, l = ax[0].get_legend_handles_labels(); legend_below(fig, h, l, ncol=len(show))
    save(fig, "fig_B_dynamics")

def fig_C():
    """Energy per token versus effective perplexity, mean and 95% CI (double column)."""
    d = runs()
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
    for a, c, l in zip(ax, (MAIN_COND, STRESS_COND), "ab"):
        dc = cond(d, c)
        for m in METHODS:
            r = dc[dc.method == m]; col, mk, _, _ = STYLE[m]
            a.errorbar(r.eff_ppl.mean(), r.j_per_tok.mean(), xerr=ci95(r.eff_ppl),
                       yerr=ci95(r.j_per_tok), fmt=mk, color=col, ms=5, capsize=2,
                       elinewidth=0.6, label=m)
        a.set_ylabel(LABEL["j_per_tok"]); tag(a, l, f"{LABEL['eff_ppl']}  [{cname(c)}]")
    fig.tight_layout(w_pad=1.2)
    h, lab = ax[0].get_legend_handles_labels(); legend_below(fig, h, lab)
    save(fig, "fig_C_pareto")

def _samples(c, kind):
    out = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "samples", "*.npz")):
        z = np.load(f, allow_pickle=True); jid = str(z["jid"]).split("|")
        if jid[0] != "main" or jid[2] != c["soc"] or float(jid[3]) != c["amb"] or jid[4] != c["wl"]:
            continue
        out.setdefault(jid[1], []).append(z[kind])
    return {m: np.concatenate(v) for m, v in out.items()}

def fig_D():
    """Response latency CDF (double column)."""
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    for a, c, l in zip(ax, (MAIN_COND, STRESS_COND), "ab"):
        s = _samples(c, "lat")
        for m in METHODS:
            x = np.sort(s[m]); col, _, ls, _ = STYLE[m]
            a.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, ls=ls, lw=1.0, label=m)
        a.set_xscale("log"); a.set_ylabel("CDF"); a.set_ylim(0, 1.0)
        tag(a, l, f"Response latency (s)  [{cname(c)}]")
    fig.tight_layout(w_pad=1.2)
    h, lab = ax[0].get_legend_handles_labels(); legend_below(fig, h, lab)
    save(fig, "fig_D_latency_cdf")

def _heat(a, M, xt, yt, fmt, cmap):
    im = a.imshow(M, cmap=cmap, aspect="auto"); a.grid(False)
    a.set_xticks(range(len(xt))); a.set_xticklabels(xt); a.set_yticks(range(len(yt)))
    a.set_yticklabels(yt)
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; dark = hi > lo and (v - lo) / (hi - lo) > 0.6
            a.text(j, i, fmt(v), ha="center", va="center", fontsize=7,
                   color="w" if dark else "k")
    return im

def fig_E():
    """Horizon x margin sensitivity on STRESS_COND, raw values (double column)."""
    d = runs("HM"); g = d.groupby(["H", "skin_margin"])
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 1.9))
    for a, (k, lab), l in zip(ax, [("throttle_s", "Throttle time (s)"),
                                   ("peak_skin", "Peak skin temp. (°C)"),
                                   ("eff_ppl", "Effective perplexity")], "abc"):
        M = g[k].mean().unstack("skin_margin").reindex(index=SWEEP_H, columns=SWEEP_M).to_numpy()
        im = _heat(a, M, [f"{m:g}" for m in SWEEP_M], [f"{h:g}" for h in SWEEP_H],
                   (lambda v: f"{v:.1f}") if k != "eff_ppl" else (lambda v: f"{v:.2f}"), "Greys")
        a.set_ylabel("Horizon $H$ (s)"); tag(a, l, f"Skin margin (°C): {lab}")
        fig.colorbar(im, ax=a, fraction=0.05, pad=0.03)
    fig.tight_layout(w_pad=0.8); save(fig, "fig_E_sensitivity")

def fig_F():
    """Model mismatch robustness on STRESS_COND (double column)."""
    d = runs("mismatch"); show = ["EdgeMoE", "ReactiveDVFS", "FiDRL", "ThermoRoute"]
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 1.9))
    for a, k, l in zip(ax, ["peak_skin", "throttle_s", "eff_ppl"], "abc"):
        for m in show:
            g = d[d.method == m].groupby("mismatch")[k]; col, mk, ls, _ = STYLE[m]
            a.errorbar(g.mean().index * 100, g.mean(), yerr=g.agg(ci95), color=col, marker=mk,
                       ls=ls, capsize=2, elinewidth=0.6, label=m)
        a.set_ylabel(LABEL[k]); tag(a, l, "Plant R/C mismatch (± %)")
    fig.tight_layout(w_pad=1.0)
    h, lab = ax[0].get_legend_handles_labels(); legend_below(fig, h, lab)
    save(fig, "fig_F_mismatch")

def fig_G():
    """Ambient stress for both SoCs, agent workload (double column, 2 x 3 bars)."""
    d = runs(); d = d[d.wl == "agent"]
    ref = d[d.method == "EdgeMoE"].groupby(["soc", "amb"]).eff_ppl.mean()
    fig, ax = plt.subplots(2, 3, figsize=(DOUBLE, 3.6), sharex=True)
    w = 0.8 / len(METHODS); x = np.arange(len(AMBIENTS))
    for i, soc in enumerate(SOCS):
        for j, (k, lab) in enumerate([("throttle_s", "Throttle time (s)"),
                                      ("ppl_inc", "Perplexity increase (%)"),
                                      ("j_per_tok", "Energy per token (J)")]):
            a = ax[i, j]
            for n, m in enumerate(METHODS):
                r = d[(d.soc == soc) & (d.method == m)].copy()
                if k == "ppl_inc":
                    r[k] = 100 * (r.eff_ppl / r.amb.map(lambda t: ref[(soc, t)]) - 1)
                g = r.groupby("amb")[k]; col, _, _, hat = STYLE[m]
                a.bar(x + (n - (len(METHODS) - 1) / 2) * w, g.mean().reindex(AMBIENTS), w,
                      yerr=g.agg(ci95).reindex(AMBIENTS), color=col, hatch=hat, ec="k", lw=0.3,
                      error_kw=dict(lw=0.5, capsize=1), label=m)
            a.set_ylabel(lab); a.set_ylim(bottom=0); a.grid(axis="x", visible=False)
            a.set_xticks(x); a.set_xticklabels([f"{t:g} °C" for t in AMBIENTS])
            if i == 1: tag(a, "abcdef"[i * 3 + j], f"Ambient, {soc}")
            else: a.set_xlabel(f"({'abcdef'[j]}) {soc}")
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    h, lab = ax[0, 0].get_legend_handles_labels(); legend_below(fig, h, lab)
    save(fig, "fig_G_ambient")

def fig_H():
    """Share of active time per expert budget, SoC-A agent, mean over seeds (single column)."""
    d = runs(); d = d[(d.soc == MAIN_COND["soc"]) & (d.wl == "agent")]
    show = ["ThermoRoute", "ThermoRoute+Spec", "Oracle"]
    cols = [f"frac_k{k}" for k in K_LEVELS]
    shades = plt.cm.Greys(np.linspace(0.15, 0.85, len(K_LEVELS)))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    pos, labels, p = [], [], 0.0
    for m in show:
        for amb in AMBIENTS:
            fr = d[(d.method == m) & (d.amb == amb)][cols].mean().to_numpy()
            bottom = 0.0
            for j, v in enumerate(fr):
                ax.bar(p, v, 0.8, bottom=bottom, color=shades[j], ec="k", lw=0.3,
                       label=f"$k={K_LEVELS[j]}$" if p == 0 else None)
                bottom += v
            pos.append(p); labels.append(f"{amb:g}"); p += 1
        p += 0.6
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    for n, m in enumerate(show):
        ax.text(np.mean(pos[n * 3:(n + 1) * 3]), -0.11, m, ha="center", va="top", fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Fraction of active time"); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    ax.set_xlabel("Ambient (°C)", labelpad=11)
    fig.tight_layout()
    h, lab = ax.get_legend_handles_labels(); legend_below(fig, h, lab)
    save(fig, "fig_H_k_occupancy")

def fig_I():
    """Per decision controller overhead on active decisions (single column)."""
    s = _samples(MAIN_COND, "ovh")
    data = [np.maximum(s[m], 1e-4) for m in METHODS]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.6),
                    capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for b, m in zip(bp["boxes"], METHODS):
        b.set_facecolor(STYLE[m][0]); b.set_hatch(STYLE[m][3]); b.set_alpha(0.8)
    ax.set_yscale("log"); ax.set_ylabel("Decision time (ms)")
    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels(METHODS, rotation=35, ha="right"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "fig_I_overhead")

def fig_J():
    """Method x metric table on STRESS_COND: raw means annotated, shade = rank in column."""
    d = cond(runs(), STRESS_COND)
    cols = [("peak_skin", "Peak skin", -1), ("throttle_s", "Throttle (s)", -1),
            ("eff_ppl", "PPL", -1), ("lat_p95", "p95 lat. (s)", -1),
            ("slo_viol_pct", "SLO viol. (%)", -1), ("j_per_tok", "J/token", -1),
            ("tok_per_charge", "Tok/charge", 1), ("responses", "Responses", 1)]
    raw = d.groupby("method")[[c for c, _, _ in cols]].mean().reindex(METHODS)
    rank = raw.copy()
    for c, _, s in cols:
        rank[c] = raw[c].rank(ascending=(s < 0), method="average")
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.5))
    ax.imshow(rank.to_numpy(), cmap="Greys", vmin=0.5, vmax=len(METHODS) + 3, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l, _ in cols])
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS); ax.grid(False)
    ax.xaxis.tick_top()
    for i in range(len(METHODS)):
        for j, (c, _, _) in enumerate(cols):
            v = raw.iat[i, j]
            txt = ("0" if v == 0 else f"{v / 1000:.0f}k" if abs(v) >= 10000 else
                   f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 1 else f"{v:.3f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    ax.set_xlabel(f"Shade: rank in column, lighter is better  [{cname(STRESS_COND)}]")
    fig.tight_layout(); save(fig, "fig_J_normalized")

def fig_K():
    """Thermal penalty weight trade off on STRESS_COND (double column)."""
    d = runs("lam"); g = d.groupby("lam")
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 1.9))
    for a, k, l in zip(ax, ["throttle_s", "eff_ppl", "j_per_tok"], "abc"):
        a.errorbar(g[k].mean().index, g[k].mean(), yerr=g[k].agg(ci95), marker="*",
                   color=STYLE["ThermoRoute"][0], capsize=2, lw=0.9, elinewidth=0.5)
        a.set_xscale("log"); a.set_ylabel(LABEL[k]); tag(a, l, r"Penalty weight $\lambda$")
    fig.tight_layout(w_pad=1.0); save(fig, "fig_K_lambda")

def make_all():
    for f in (fig_A, fig_B, fig_C, fig_D, fig_E, fig_F, fig_G, fig_H, fig_I, fig_J, fig_K):
        f()
    print("figures in", FIG_DIR)
