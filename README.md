# ThermoRoute: Thermal Throttling Aware Expert Scheduling for Smartphone Mixture of Experts Inference

Code, data, and results for the ThermoRoute manuscript (under review). ThermoRoute is a model predictive controller that jointly selects the active expert budget $k$, the DVFS frequency level, and the processor cluster of a mixture of experts (MoE) language model running on a smartphone, using a filtered, bias corrected two node (die and skin) thermal model with a closed form 180 s skin lookahead.

## Contents

1. [Overview](#overview)
2. [What is measured, derived, and assumed](#what-is-measured-derived-and-assumed)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Reproducing the results](#reproducing-the-results)
6. [Main results](#main-results)
7. [Figures](#figures)
8. [Data sources and licences](#data-sources-and-licences)
9. [Limitations](#limitations)
10. [Citation and licence](#citation-and-licence)

## Overview

Sustained MoE decoding heats a phone until the operating system governor throttles it, and throughput then oscillates between fast and very slow. ThermoRoute predicts the die and skin temperature trajectory for each of the 48 candidate actions $(k, f, c)$ with $k \in \{2,4,6,8\}$, six frequency levels, and two clusters, and applies the lowest cost action every second. The cost combines measured model quality, decoding speed, energy per token, soft thermal penalties over a 10 s horizon, a terminal skin lookahead penalty, and a switching cost.

The controller is compared against EdgeMoE, EdgeLLM (speculative decoding), a static $k = 4$ budget, reactive DVFS, a tabular Q learning DVFS agent (FiDRL), ThermoRoute with speculative decoding, and an Oracle with the true plant parameters. All methods share the same plant, governor, and workload, over two SoC classes, three ambient temperatures (25, 30, 35 °C), two workloads (chat and agent), and 20 test seeds disjoint from 20 training seeds.

## What is measured, derived, and assumed

| Category | Quantity |
|---|---|
| Measured | OLMoE 1B 7B perplexity for every $k$ on WikiText 103, distinct experts per layer over consecutive tokens from router logits, parameter counts from the model config |
| Real | Prompt and response token lengths of LMSYS Chat 1M conversations, tokenized with the OLMoE tokenizer |
| Derived | Phone decoding rate from a memory bound model (achieved bandwidth divided by bytes per token) |
| Representative | Thermal resistances and capacitances, power coefficients, governor thresholds, SoC profiles |
| Assumed | Think time, prefill speedup, speculative acceptance rate, speed target, SLO rate |

This is a calibrated simulation of two SoC classes, not a measurement campaign on commercial handsets.

## Repository layout

```
ThermoRoute/
├── ThermoRoute.ipynb            # end to end pipeline, Steps 0 to 7 (Colab or Jupyter)
├── sim/
│   ├── config.py                # single source of every constant (SoC, governor, MPC, sweeps)
│   ├── model_profile.py         # OLMoE parameter counts, measured quality and expert union
│   ├── plant.py                 # bandwidth, power, two node RC thermal plant, OS governor
│   ├── workload.py              # closed loop single user workload from LMSYS lengths
│   ├── controllers.py           # ThermoRoute MPC and all baselines
│   ├── runner.py                # episodes, FiDRL training, weight tuning, experiment grid
│   ├── stats_tables.py          # means, 95% t intervals, Wilcoxon with Holm, LaTeX tables
│   └── figs.py                  # all result figures
├── data/
│   ├── k_profile.csv            # measured perplexity and 95% bootstrap interval per k
│   ├── expert_union.csv         # measured distinct experts per layer, windows 1 to 5
│   ├── olmoe_config.json        # OLMoE 1B 7B configuration
│   └── requests.csv             # 9456 request pairs (token counts only, no text)
├── results/
│   ├── all_runs.csv             # one row per simulated session
│   ├── summary_all_conditions.csv
│   ├── wilcoxon_main.csv, wilcoxon_stress.csv
│   ├── table_main.tex, table_stress.tex
│   ├── mpc_tuning.csv, mpc_weights.json   # weight grid search on training seeds
│   ├── fidrl_tables.pkl         # trained FiDRL Q tables
│   └── raw_runs.zip             # per session samples and traces used by figures B, D and I
├── figures/                     # final figures, PNG 900 dpi
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.10 or later is recommended; the archived results were produced with Python 3.13 and on Google Colab.

```bash
git clone https://github.com/<jingum741>/ThermoRoute.git
cd ThermoRoute
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Only `numpy`, `scipy`, `pandas`, and `matplotlib` are needed to rerun the simulator and redraw figures. The remaining packages are needed only for Steps 1 and 2. `transformers` is pinned to 4.51.3 so that OLMoE experts stay `nn.Linear` layers for 8 bit loading. Figures use Times New Roman; on Linux install `ttf-mscorefonts-installer`.

## Reproducing the results

Open `ThermoRoute.ipynb` from the repository root and run the cells from top to bottom. Each step saves its outputs and is skipped on a rerun.

| Step | Output | Hardware | Approx. time |
|---|---|---|---|
| 0 | settings, packages, fonts, Hugging Face login | any | 3 to 5 min |
| 1 | measured OLMoE perplexity per $k$, router expert union | GPU (Colab T4) | 20 to 40 min |
| 2 | LMSYS Chat 1M request lengths | any | 5 to 15 min |
| 3 | simulator sanity checks | CPU | 1 min |
| 4 | MPC weight tuning, FiDRL training, full experiment grid | CPU | 30 to 60 min |
| 5 | LaTeX tables, Wilcoxon tests with Holm correction | CPU | 1 min |
| 6 | figures (PNG and PDF) | CPU | 2 min |
| 7 | claim check against the abstract, results archive | CPU | 1 min |

Steps 1 and 2 can be skipped because their outputs are already in `data/`. Step 2 requires accepting the LMSYS Chat 1M terms on Hugging Face and a read token.

To regenerate the tables and figures from the archived runs without rerunning the grid:

```bash
cd results && unzip raw_runs.zip && cd ..
export TR_ROOT=$PWD TR_FIG_DIR=$PWD/figures_regenerated   # Windows: set TR_ROOT=%cd%
cd sim
python -c "import stats_tables; stats_tables.main()"
python -c "import figs; figs.make_all()"
```

Step 6b of the notebook contains the final versions of Figs. A, C, and H used in the manuscript.

## Main results

SoC A, agent workload, 35 °C ambient, mean over 20 test seeds (from `results/summary_all_conditions.csv`).

| Method | Throttle (s) | PPL<sub>eff</sub> | p95 latency (s) | Energy per token (J) | Tokens per charge (×10³) |
|---|---:|---:|---:|---:|---:|
| EdgeMoE | 1358.1 | 8.93 | 28.80 | 0.1436 | 482.8 |
| EdgeLLM | 1413.6 | 8.93 | 29.46 | 0.1483 | 467.7 |
| ReactiveDVFS | 9.4 | 8.93 | 28.50 | 0.1251 | 554.3 |
| FiDRL | 849.4 | 8.93 | 26.20 | 0.1318 | 526.1 |
| StaticK4 | 1299.2 | 11.53 | 19.09 | 0.0978 | 708.9 |
| **ThermoRoute** | **63.0** | 9.40 | **17.99** | **0.0969** | **715.5** |
| ThermoRoute+Spec | 90.1 | 9.40 | 18.23 | 0.0981 | 706.3 |
| Oracle | 102.3 | 9.40 | 17.89 | 0.0975 | 710.8 |

Measured OLMoE 1B 7B profile (`data/k_profile.csv`): perplexity 27.38, 11.53, 9.40, and 8.93 for $k$ = 2, 4, 6, and 8.

## Figures

| File | Manuscript figure |
|---|---|
| `fig_A_k_profile.png` | Fig. 5, measured model profile |
| `fig_B_dynamics.png` | Fig. 6, temporal dynamics at 35 °C |
| `fig_C_pareto.png` | Fig. 7, energy per token versus effective perplexity |
| `fig_D_latency_cdf.png` | Fig. 8, response latency CDF |
| `fig_G_ambient.png` | Fig. 9, ambient temperature and SoC class |
| `fig_H_k_occupancy.png` | Fig. 10, expert budget occupancy |
| `fig_E_sensitivity.png` | Fig. 11, horizon and skin margin sensitivity |
| `fig_K_lambda.png` | Fig. 12, penalty weight |
| `fig_F_mismatch.png` | Fig. 13, plant mismatch robustness |
| `fig_I_overhead.png`, `fig_J_normalized.png` | supplementary (controller overhead, normalized comparison) |

## Data sources and licences

1. OLMoE 1B 7B (`allenai/OLMoE-1B-7B-0924`), Muennighoff et al., ICLR 2025.
2. WikiText 103 (`Salesforce/wikitext`, `wikitext-103-raw-v1`), Merity et al., ICLR 2017.
3. LMSYS Chat 1M (`lmsys/lmsys-chat-1m`), Zheng et al., ICLR 2024. The dataset is gated and its terms restrict redistribution. `data/requests.csv` contains only conversation identifiers, serving model names, and token counts; no conversation text is included. Users who need the text must obtain it from the original source under its terms.

## Limitations

The thermal and power parameters are representative values for two SoC classes rather than parameters identified on a specific handset. Response lengths come from the models that served LMSYS rather than from OLMoE. The controller assumes a single foreground model, and plant mismatch above about 20% erodes the throttling advantage.

## Citation and licence

A citation will be added once the manuscript is published. The code is released under the MIT licence (see `LICENSE`). Data files retain the licences of their original sources.
