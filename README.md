# XZD-Sentinel

Code and results for **"XZD-Sentinel: Disjunctive Fusion of Closed-Set and Open-Set Detectors with SHAP-Based Explanations for Zero-Day Intrusion Detection"** (Thakar and Popat).
Branch A = XGBoost, B = MLP, C = Isolation Forest trained on benign traffic only; the proposed rule is the OR of the branch decisions.

## Data
NSL-KDD is not redistributed here. Download `KDDTrain+.txt` and `KDDTest+.txt` (125,973 and 22,544 records) from
https://github.com/defcom17/NSL_KDD and place them in `data/`. The 17 attack subtypes of KDDTest+ that are absent from KDDTrain+ define the zero-day subset (listed in the paper, Section 4.1).

## Environment
Python 3.13, see `requirements.txt`. Random seeds 42-51 are used for the ten-seed experiments.

## Main experiments (paper Sections 5.1-5.4, 5.7-5.10)
    python xzd_experiments.py --train data/KDDTrain+.txt --test data/KDDTest+.txt --seeds 10 --out results_main
Writes the `exp1`-`exp5` tables (matched false-alarm rate, McNemar tests, branch ablation, ten-seed summary, per-subtype and per-family recall) and the ROC figure. The outputs are in `results_main/`.

## Additional experiments (`additional_experiments/`)
Run from inside that folder. The data folder is `../data/` by default; set the environment variable `XZD_DATA` to use another location.

| Script | Output | Paper |
|---|---|---|
| `main_run.py`, then `ae_rerun.py` | `main.pkl` (Isolation Forest, autoencoder and Mahalanobis scores, ten seeds) | Sections 4.3, 5.12 |
| `lofo_run.py`, `an_lofo.py` | `lofo_results.txt` | Sections 4.8, 5.9, Table 13 |
| `tune.py`, `tune_ae.py` | `tune.log` (detector hyper-parameters, chosen on training data only) | Section 4.3 |
| `an_cmp.py` | `cmp_results.txt` (alternative detectors, subtype-level bootstrap) | Sections 5.11-5.12, Table 16 |
| `calib.py` | `calib_results.txt` (learned logistic fusion) | Section 5.12 |
| `an_matched_cluster.py` | subtype-level interval of the gain at matched false-alarm rate | Section 5.11 |
| `del_test.py` | `del_results.txt` (deletion test) | Section 5.6, Table 9 |

## Notes
* Results were produced on CPU only. Latency figures in the paper are single-threaded medians on a 2-vCPU server and indicate order of magnitude only.
* License: MIT.
