# xzd_experiments.py  --  reviewer-requested experiments for XZD-Sentinel (JISA submission)
#
# Covers
#   1. Recall at matched false-positive rate, FPR table, OR-combinations of baselines
#   2. McNemar tests on the FULL test set (and on subsets), incl. proposed vs. a no-open-set OR
#   3. Pairwise branch ablation (A|B, A|C, B|C, A|B|C) + unique-detection (Venn) counts
#   4. Multi-seed runs (mean +/- SD, paired comparisons across seeds)
#   5. Per-subtype recall for the held-out zero-day attack subtypes
#
# Branch naming:  A = XGBoost, B = MLP, C = Isolation Forest (trained on benign only).
# The proposed rule is the disjunction A|B|C at each branch's native threshold.
#
# Usage
#   python xzd_experiments.py --train KDDTrain+.txt --test KDDTest+.txt --seeds 10 --out results
#   python xzd_experiments.py ... --with-stacking      # also run the stacking ablation (slower)
#   python xzd_experiments.py ... --fast               # tiny models, ONLY for a quick smoke test
#
# Requires: numpy pandas scipy scikit-learn xgboost (matplotlib optional, for the ROC figure)
#
# IMPORTANT: edit the CONFIG block so it matches your original notebook, then check the
# "REPRODUCTION CHECK" printed for the first seed against Tables 2-4 of the paper.
import argparse, json, os, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =============================== CONFIG (EDIT ME) ===============================
RF_KW = dict(n_estimators=300, n_jobs=-1)
XGB_KW = dict(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss")
MLP_KW = dict(hidden_layer_sizes=(128, 64), max_iter=200)      # <-- set to your original MLP settings
IF_KW = dict(n_estimators=300, n_jobs=-1)                      # <-- add contamination / max_samples if used
IF_PERCENTILE = 90          # threshold = 90th percentile of training novelty scores
IF_TAU_FROM = "all_train"   # "all_train" or "benign_train"; pick the one that reproduces Table 2
                            # (Table 2 shows IF precision 99.86%, i.e. almost no false alarms, which
                            #  suggests the percentile is taken over ALL training rows)
TAU_CLASSIFIER = 0.5        # native threshold for RF / XGBoost / MLP
FPR_GRID = [0.005, 0.01, 0.02, 0.05, 0.10]
SEED0 = 42                  # primary seed (the one used in the paper)
# ================================================================================

FEATURES = ("duration protocol_type service flag src_bytes dst_bytes land wrong_fragment urgent hot "
            "num_failed_logins logged_in num_compromised root_shell su_attempted num_root "
            "num_file_creations num_shells num_access_files num_outbound_cmds is_host_login "
            "is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate "
            "same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count "
            "dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate "
            "dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate "
            "dst_host_rerror_rate dst_host_srv_rerror_rate").split()
assert len(FEATURES) == 41
CAT = ["protocol_type", "service", "flag"]
NUM = [f for f in FEATURES if f not in CAT]

PAPER_ZERO_DAY = ["apache2", "httptunnel", "mailbomb", "mscan", "named", "processtable", "ps", "saint",
                  "sendmail", "snmpgetattack", "snmpguess", "sqlattack", "udpstorm", "worm", "xlock",
                  "xsnoop", "xterm"]
FAMILY = {"apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
          "mscan": "Probe", "saint": "Probe",
          "httptunnel": "R2L", "named": "R2L", "sendmail": "R2L", "snmpgetattack": "R2L",
          "snmpguess": "R2L", "xlock": "R2L", "xsnoop": "R2L",
          "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R"}

# Values reported in the paper (percent) for the reproduction check: acc, recall, zero-day recall, AUC
PAPER = {"RF": (77.67, 62.79, 28.59, 96.19), "A (XGBoost)": (79.00, 65.25, 36.27, 96.40),
         "B (MLP)": (80.68, 68.33, 48.59, 90.26), "C (IsolationForest)": (49.32, 10.99, 5.04, 94.40),
         "A|B|C (proposed)": (81.99, 70.90, 50.75, 94.74), "Stacking (ablation)": (79.64, 66.41, 40.13, 95.18)}

CONFIGS = {
    "RF": ["RF"], "A (XGBoost)": ["XGB"], "B (MLP)": ["MLP"], "C (IsolationForest)": ["IF"],
    "A|B": ["XGB", "MLP"], "A|C": ["XGB", "IF"], "B|C": ["MLP", "IF"],
    "A|B|C (proposed)": ["XGB", "MLP", "IF"],
    "RF|B": ["RF", "MLP"], "RF|A|B (no open-set)": ["RF", "XGB", "MLP"], "RF|A|B|C": ["RF", "XGB", "MLP", "IF"],
}
PROPOSED = "A|B|C (proposed)"
SINGLE_CLASSIFIERS = ["RF", "A (XGBoost)", "B (MLP)"]
BEST_SINGLE = SINGLE_CLASSIFIERS + ["C (IsolationForest)"]   # single detectors used for "best single" comparisons
ALL_COMPARATORS = BEST_SINGLE + ["A|B", "A|C", "A|B|C (proposed)", "RF|A|B (no open-set)"]
BRANCH_LABEL = {0: "none", 1: "A only", 2: "B only", 3: "A+B", 4: "C only", 5: "A+C", 6: "B+C", 7: "A+B+C"}


# ------------------------------------------------------------------ data
def load_nsl(path):
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 43:
        df.columns = FEATURES + ["label", "difficulty"]
    elif df.shape[1] == 42:
        df.columns = FEATURES + ["label"]
    else:
        raise ValueError(f"{path}: expected 42 or 43 columns, found {df.shape[1]}")
    df["label"] = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
    return df


def prepare(train_path, test_path):
    tr, te = load_nsl(train_path), load_nsl(test_path)
    ytr = (tr["label"] != "normal").astype(int).values
    yte = (te["label"] != "normal").astype(int).values
    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", StandardScaler(), NUM)])
    Xtr = ct.fit_transform(tr[FEATURES]).astype(np.float32)   # fit on TRAIN only
    Xte = ct.transform(te[FEATURES]).astype(np.float32)
    unseen = sorted(set(te["label"]) - set(tr["label"]) - {"normal"})
    zd = te["label"].isin(unseen).values
    print(f"features after encoding: {Xtr.shape[1]} | train {len(tr)} | test {len(te)} "
          f"| benign test {(yte == 0).sum()} | zero-day test {zd.sum()} | subtypes {len(unseen)}")
    if unseen != sorted(PAPER_ZERO_DAY) or zd.sum() != 3750:
        print("WARNING: held-out subtypes differ from the paper (expected 17 subtypes, n=3750).")
        print("         found:", unseen)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, zd=zd, subtype=te["label"].values, unseen=unseen)


# ------------------------------------------------------------------ models
def novelty(iso, X):
    return -iso.score_samples(X)          # higher = more anomalous


def fit_scores(D, seed, with_stacking):
    Xtr, ytr, Xte = D["Xtr"], D["ytr"], D["Xte"]
    rf = RandomForestClassifier(random_state=seed, **RF_KW).fit(Xtr, ytr)
    xgb = XGBClassifier(random_state=seed, **XGB_KW).fit(Xtr, ytr)
    mlp = MLPClassifier(random_state=seed, **MLP_KW).fit(Xtr, ytr)
    iso = IsolationForest(random_state=seed, **IF_KW).fit(Xtr[ytr == 0])
    ref = Xtr if IF_TAU_FROM == "all_train" else Xtr[ytr == 0]
    tau_c = float(np.percentile(novelty(iso, ref), IF_PERCENTILE))
    assert tau_c > 0, "novelty threshold must be positive for the max-normalised fusion score"
    scores = {"RF": rf.predict_proba(Xte)[:, 1], "XGB": xgb.predict_proba(Xte)[:, 1],
              "MLP": mlp.predict_proba(Xte)[:, 1], "IF": novelty(iso, Xte)}
    tau = {"RF": TAU_CLASSIFIER, "XGB": TAU_CLASSIFIER, "MLP": TAU_CLASSIFIER, "IF": tau_c}
    if with_stacking:
        scores["STACK"], tau["STACK"] = fit_stacking(D, seed, scores)
    dec = {k: scores[k] >= tau[k] for k in scores}
    return dict(seed=seed, scores=scores, tau=tau, dec=dec)


def fit_stacking(D, seed, test_scores):
    """OOF predictions of A, B, C -> logistic regression; threshold tuned for F1 on a 20% split."""
    Xtr, ytr = D["Xtr"], D["ytr"]
    oof = np.zeros((len(ytr), 3))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr):
        a = XGBClassifier(random_state=seed, **XGB_KW).fit(Xtr[tr], ytr[tr])
        b = MLPClassifier(random_state=seed, **MLP_KW).fit(Xtr[tr], ytr[tr])
        c = IsolationForest(random_state=seed, **IF_KW).fit(Xtr[tr][ytr[tr] == 0])
        oof[va] = np.c_[a.predict_proba(Xtr[va])[:, 1], b.predict_proba(Xtr[va])[:, 1], novelty(c, Xtr[va])]
    feats = lambda m: np.c_[m, np.abs(m[:, 0] - m[:, 1])]       # + confidence-gap feature
    M = feats(oof)
    Mt, Mv, yt, yv = train_test_split(M, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pv = mk().fit(Mt, yt).predict_proba(Mv)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    thr = float(grid[np.argmax([f1_score(yv, pv >= t) for t in grid])])
    lr = mk().fit(M, ytr)
    Mte = feats(np.c_[test_scores["XGB"], test_scores["MLP"], test_scores["IF"]])
    return lr.predict_proba(Mte)[:, 1], thr


def config_arrays(run, name, configs):
    mem = configs[name]
    score = np.max([run["scores"][m] / run["tau"][m] for m in mem], axis=0)   # >=1  <=>  OR rule fires
    dec = np.logical_or.reduce([run["dec"][m] for m in mem])
    return score, dec


# ------------------------------------------------------------------ metrics
def tpr_at_fpr(pos, benign, score, f):
    """TPR of `pos` rows at false-positive rate f on `benign` rows (ROC-interpolated; handles ties)."""
    m = pos | benign
    fpr, tpr, _ = roc_curve(pos[m].astype(int), score[m])
    return float(np.interp(f, fpr, tpr))


def basic_metrics(y, pred, zd):
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    known = ~zd
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / len(y), prec=prec, rec=rec, f1=2 * prec * rec / max(prec + rec, 1e-12),
                fpr=fp / max(fp + tn, 1), fp=fp, zd_recall=float(pred[zd].mean()),
                known_attack_recall=float(pred[known & (y == 1)].mean()),
                acc_known=float((pred[known] == y[known]).mean()))


def evaluate_run(run, D, configs):
    y, zd = D["yte"], D["zd"]; benign = y == 0
    rows = {}
    for name in configs:
        score, pred = config_arrays(run, name, configs)
        r = basic_metrics(y, pred, zd)
        r["auc"] = roc_auc_score(y, score)
        m = zd | benign
        r["auc_zd_vs_benign"] = roc_auc_score(zd[m].astype(int), score[m])
        rows[name] = r
    f_prop = rows[PROPOSED]["fpr"]
    for name in configs:                   # recall of every config at the FPR of the proposed rule
        score, _ = config_arrays(run, name, configs)
        rows[name]["zd_tpr_at_prop_fpr"] = tpr_at_fpr(zd, benign, score, f_prop)
        rows[name]["all_tpr_at_prop_fpr"] = tpr_at_fpr(y == 1, benign, score, f_prop)
    return pd.DataFrame(rows).T


def subtype_recall(run, D, configs, names):
    out = []
    for st in D["unseen"]:
        m = D["subtype"] == st
        row = {"subtype": st, "family": FAMILY.get(st, "?"), "n": int(m.sum())}
        for nme in names:
            row[nme] = float(config_arrays(run, nme, configs)[1][m].mean())
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ statistics
def mcnemar(y, base_pred, prop_pred, mask):
    cb, cp = (base_pred == y)[mask], (prop_pred == y)[mask]
    n01 = int((~cb & cp).sum())    # baseline wrong, proposed right
    n10 = int((cb & ~cp).sum())    # baseline right, proposed wrong
    n = n01 + n10
    chi2 = (abs(n01 - n10) - 1) ** 2 / n if n else 0.0
    p_chi = float(stats.chi2.sf(chi2, 1)) if n else 1.0
    p_exact = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue) if n else 1.0
    return dict(n01=n01, n10=n10, chi2=chi2, p_chi2=p_chi, p_exact=p_exact)


def venn_counts(run, mask):
    code = (run["dec"]["XGB"].astype(int) + 2 * run["dec"]["MLP"].astype(int) + 4 * run["dec"]["IF"].astype(int))[mask]
    return {BRANCH_LABEL[k]: int((code == k).sum()) for k in range(8)}


# ------------------------------------------------------------------ analyses on the primary seed
def reproduction_check(df):
    print("\n=== REPRODUCTION CHECK (seed %d) : yours vs. paper (percentage points) ===" % SEED0)
    for name, (acc, rec, zdr, auc) in PAPER.items():
        if name not in df.index: continue
        r = df.loc[name]
        d = [100 * r["acc"] - acc, 100 * r["rec"] - rec, 100 * r["zd_recall"] - zdr, 100 * r["auc"] - auc]
        flag = "" if max(abs(x) for x in d) < 1.0 else "   <-- differs: check CONFIG"
        print(f"{name:22s} acc {100*r['acc']:6.2f} ({d[0]:+5.2f})  rec {100*r['rec']:6.2f} ({d[1]:+5.2f})  "
              f"zd {100*r['zd_recall']:6.2f} ({d[2]:+5.2f})  auc {100*r['auc']:6.2f} ({d[3]:+5.2f}){flag}")


def exp1_matched_fpr(run, D, configs, out):
    y, zd = D["yte"], D["zd"]; benign = y == 0
    f_prop = basic_metrics(y, config_arrays(run, PROPOSED, configs)[1], zd)["fpr"]
    grid = FPR_GRID + [f_prop]
    rows = []
    for name in configs:
        score, pred = config_arrays(run, name, configs)
        r = {"config": name, "native_FPR": basic_metrics(y, pred, zd)["fpr"],
             "native_zd_recall": float(pred[zd].mean())}
        for f in grid:
            tag = "prop_FPR(%.2f%%)" % (100 * f_prop) if f == f_prop else "FPR=%.1f%%" % (100 * f)
            r["zd_recall@" + tag] = tpr_at_fpr(zd, benign, score, f)
        r["AUC_zd_vs_benign"] = roc_auc_score(zd[zd | benign].astype(int), score[zd | benign])
        rows.append(r)
    t = pd.DataFrame(rows).set_index("config")
    t.to_csv(f"{out}/exp1_matched_fpr.csv")
    # gain of the proposed rule over the best single classifier at each matched FPR
    gains = []
    for f in grid:
        sc = {n: tpr_at_fpr(zd, benign, config_arrays(run, n, configs)[0], f) for n in BEST_SINGLE + [PROPOSED]}
        best = max(BEST_SINGLE, key=lambda n: sc[n])
        gains.append(dict(FPR=f, proposed=sc[PROPOSED], best_single=best, best_single_recall=sc[best],
                          gain_pp=100 * (sc[PROPOSED] - sc[best])))
    g = pd.DataFrame(gains); g.to_csv(f"{out}/exp1_gain_over_best_single.csv", index=False)
    print("\n=== EXP 1: zero-day recall at matched FPR (ROC-interpolated; FPR measured on benign test traffic) ===")
    print(t.round(4).to_string()); print("\nGain of proposed over the best single classifier at the same FPR:")
    print(g.round(4).to_string(index=False))


def exp2_mcnemar(run, D, configs, out):
    y, zd = D["yte"], D["zd"]; known = ~zd
    prop = config_arrays(run, PROPOSED, configs)[1]
    subsets = {"full test": np.ones(len(y), bool), "zero-day": zd, "known (benign+known attacks)": known,
               "benign only": y == 0, "known attacks only": known & (y == 1)}
    rows = []
    for base in [b for b in ALL_COMPARATORS if b != PROPOSED]:
        bp = config_arrays(run, base, configs)[1]
        for sname, mask in subsets.items():
            rows.append(dict(baseline=base, subset=sname, **mcnemar(y, bp, prop, mask)))
    t = pd.DataFrame(rows); t.to_csv(f"{out}/exp2_mcnemar.csv", index=False)
    print("\n=== EXP 2: McNemar, proposed vs baseline (n01 = baseline wrong & proposed right; n10 = reverse) ===")
    print("Note: on the zero-day subset n10 is 0 for any baseline that is a branch of the OR rule, by construction.")
    print(t.round(6).to_string(index=False))


def exp3_ablation(run, D, configs, out):
    y, zd = D["yte"], D["zd"]
    abl = ["A (XGBoost)", "B (MLP)", "C (IsolationForest)", "A|B", "A|C", "B|C", "A|B|C (proposed)"]
    ev = evaluate_run(run, D, configs).loc[abl]
    ev.to_csv(f"{out}/exp3_ablation.csv")
    venn = pd.DataFrame({"zero-day attacks": venn_counts(run, zd), "known attacks": venn_counts(run, ~zd & (y == 1)),
                         "benign (false alarms)": venn_counts(run, y == 0)})
    venn.to_csv(f"{out}/exp3_unique_detections.csv")
    print("\n=== EXP 3: pairwise ablation ==="); print(ev.round(4).to_string())
    print("\nWhich branches fire (counts). 'C only' on zero-day = detections only the Isolation Forest provides:")
    print(venn.to_string())


def exp5_subtypes(run, D, configs, out, names):
    t = subtype_recall(run, D, configs, names); t.to_csv(f"{out}/exp5_subtype_recall_seed%d.csv" % run["seed"], index=False)
    fam = t.assign(**{n: t[n] * t["n"] for n in names}).groupby("family")[["n"] + names].sum()
    for n in names: fam[n] = fam[n] / fam["n"]
    fam.to_csv(f"{out}/exp5_family_recall_seed%d.csv" % run["seed"])
    print("\n=== EXP 5: per-subtype zero-day recall (native thresholds) ==="); print(t.round(3).to_string(index=False))
    print("\nBy attack family:"); print(fam.round(3).to_string())


def save_roc_figure(run, D, configs, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping ROC figure"); return
    y, zd = D["yte"], D["zd"]
    names = list(dict.fromkeys(["RF", "A (XGBoost)", "B (MLP)", "C (IsolationForest)", "A|C", "A|B|C (proposed)"] + (["Stacking (ablation)"] if "Stacking (ablation)" in configs else [])))
    sty = {"RF": ("#1f77b4", "-"), "A (XGBoost)": ("#ff7f0e", "--"), "B (MLP)": ("#2ca02c", "-."),
           "C (IsolationForest)": ("#7f7f7f", ":"), "A|B|C (proposed)": ("#d62728", "-"), "A|C": ("#17becf", "-"), "Stacking (ablation)": ("#9467bd", "--")}
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for n in names:
        s, p = config_arrays(run, n, configs)
        fpr, tpr, _ = roc_curve(y, s); c, ls = sty[n]
        ax.plot(fpr, tpr, color=c, ls=ls, lw=2.4 if n == PROPOSED else 1.5, label=f"{n} (AUC = {100*roc_auc_score(y, s):.2f}%)")
        m = basic_metrics(y, p, zd)
        ax.scatter(m["fpr"], m["rec"], color=c, s=40, edgecolor="k", zorder=5)
    ax.plot([0, 1], [0, 1], "k:", lw=.8); ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax.grid(alpha=.3); ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
    fig.savefig(f"{out}/figure4_roc_all_configs.png", dpi=300); fig.savefig(f"{out}/figure4_roc_all_configs.pdf")


def mlp_scan(D, iters):
    """Fit the MLP with several settings (seed 42) and compare with the paper's MLP row."""
    print("Paper MLP target: acc 80.68  recall 68.33  zero-day recall 48.59  AUC 90.26")
    grid = [dict(max_iter=i, early_stopping=False) for i in iters] + [dict(max_iter=200, early_stopping=True)]
    for kw in grid:
        t0 = time.time(); base = dict(MLP_KW); base.update(kw)
        m = MLPClassifier(random_state=SEED0, **base).fit(D["Xtr"], D["ytr"])
        p = m.predict_proba(D["Xte"])[:, 1]; d = p >= TAU_CLASSIFIER; y = D["yte"]
        print(f"max_iter={kw['max_iter']:>4} early_stopping={str(kw['early_stopping']):5s} epochs_run={m.n_iter_:>3} | "
              f"acc {100*(d==y).mean():5.2f}  recall {100*d[y==1].mean():5.2f}  zd-recall {100*d[D['zd']].mean():5.2f}  "
              f"AUC {100*roc_auc_score(y, p):5.2f}  ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--out", default="results")
    ap.add_argument("--with-stacking", action="store_true"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--if-tau-from", choices=["all_train", "benign_train"], default=None,
                    help="reference set for the Isolation Forest threshold (default: value in CONFIG)")
    ap.add_argument("--if-percentile", type=float, default=None,
                    help="percentile of the reference novelty scores used as threshold (default: CONFIG)")
    ap.add_argument("--mlp-max-iter", type=int, default=None); ap.add_argument("--mlp-early-stopping", action="store_true")
    ap.add_argument("--proposed", default=None, help='configuration treated as "proposed" for matched-FPR / McNemar / paired tests, e.g. "A|C"')
    ap.add_argument("--mlp-scan", action="store_true", help="try several MLP settings on seed 42 and exit")
    ap.add_argument("--mlp-scan-iters", default="10,20,30,50,100")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    global IF_TAU_FROM, IF_PERCENTILE, PROPOSED
    if a.proposed:
        assert a.proposed in CONFIGS, f"unknown config {a.proposed}; choose from {list(CONFIGS)}"
        PROPOSED = a.proposed
    if a.if_tau_from: IF_TAU_FROM = a.if_tau_from
    if a.if_percentile is not None: IF_PERCENTILE = a.if_percentile
    if a.mlp_max_iter: MLP_KW.update(max_iter=a.mlp_max_iter)
    if a.mlp_early_stopping: MLP_KW.update(early_stopping=True)
    print(f"Proposed configuration for tests: {PROPOSED}")
    print(f"Isolation Forest threshold: percentile {IF_PERCENTILE} of {IF_TAU_FROM} novelty scores | MLP settings: {MLP_KW}")
    if a.fast:
        RF_KW.update(n_estimators=40); XGB_KW.update(n_estimators=40); IF_KW.update(n_estimators=40); MLP_KW.update(max_iter=25)
        print("FAST MODE: tiny models, results are NOT meaningful")
    configs = dict(CONFIGS)
    if a.with_stacking: configs["Stacking (ablation)"] = ["STACK"]
    D = prepare(a.train, a.test)
    if a.mlp_scan:
        mlp_scan(D, [int(v) for v in a.mlp_scan_iters.split(",")]); return
    seeds = [SEED0 + i for i in range(a.seeds)]
    per_seed, sub_rows = [], []
    sub_names = ["RF", "A (XGBoost)", "B (MLP)", "C (IsolationForest)", PROPOSED]
    for i, seed in enumerate(seeds):
        t0 = time.time(); run = fit_scores(D, seed, a.with_stacking)
        df = evaluate_run(run, D, configs); df["seed"] = seed; per_seed.append(df.reset_index().rename(columns={"index": "config"}))
        st = subtype_recall(run, D, configs, sub_names); st["seed"] = seed; sub_rows.append(st)
        print(f"seed {seed} done in {time.time()-t0:.0f}s | proposed zd-recall {df.loc[PROPOSED,'zd_recall']:.4f} "
              f"| MLP {df.loc['B (MLP)','zd_recall']:.4f} | proposed FPR {df.loc[PROPOSED,'fpr']:.4f}")
        if i == 0:                                  # detailed analyses on the primary seed
            reproduction_check(df)
            np.savez_compressed(f"{a.out}/scores_seed{seed}.npz", y=D["yte"], zero_day=D["zd"], subtype=D["subtype"],
                                tau=json.dumps(run["tau"]), **{f"score_{k}": v for k, v in run["scores"].items()})
            exp1_matched_fpr(run, D, configs, a.out); exp2_mcnemar(run, D, configs, a.out)
            exp3_ablation(run, D, configs, a.out); exp5_subtypes(run, D, configs, a.out, sub_names)
            save_roc_figure(run, D, configs, a.out)

    # ---------------- Exp 4: multi-seed summary
    L = pd.concat(per_seed); L.to_csv(f"{a.out}/exp4_all_seeds_long.csv", index=False)
    cols = ["acc", "prec", "rec", "f1", "fpr", "zd_recall", "known_attack_recall", "auc", "auc_zd_vs_benign", "zd_tpr_at_prop_fpr"]
    g = L.groupby("config", sort=False)[cols].agg(["mean", "std"])
    summ = pd.DataFrame({c: [f"{100*g.loc[n,(c,'mean')]:.2f} ± {100*g.loc[n,(c,'std')]:.2f}" for n in g.index] for c in cols}, index=g.index)
    summ.to_csv(f"{a.out}/exp4_multiseed_summary.csv")
    print(f"\n=== EXP 4: mean ± SD over {len(seeds)} seeds (percent) ==="); print(summ.to_string())

    P = L.pivot(index="seed", columns="config")
    prow = []
    for base in [b for b in ALL_COMPARATORS if b != PROPOSED]:
        for met in ["f1", "acc", "prec", "fpr", "zd_recall", "known_attack_recall", "zd_tpr_at_prop_fpr"]:
            d = (P[(met, PROPOSED)] - P[(met, base)]).values
            mean, sd = d.mean(), d.std(ddof=1) if len(d) > 1 else float("nan")
            ci = stats.t.interval(0.95, len(d) - 1, loc=mean, scale=stats.sem(d)) if len(d) > 2 and sd > 0 else (np.nan, np.nan)
            try: pw = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) and len(d) >= 5 else np.nan
            except Exception: pw = np.nan
            prow.append(dict(proposed_minus=base, metric=met, mean_diff_pp=100 * mean, sd_pp=100 * sd,
                             ci95_lo_pp=100 * ci[0], ci95_hi_pp=100 * ci[1], seeds_proposed_better=int((d > 0).sum()),
                             n_seeds=len(d), wilcoxon_p=pw))
    pt = pd.DataFrame(prow); pt.to_csv(f"{a.out}/exp4_paired_vs_baselines.csv", index=False)
    print("\nPaired differences across seeds (proposed minus baseline, percentage points).")
    print("zd_tpr_at_prop_fpr = zero-day recall of each model when its threshold is tuned to the proposed rule's FPR.")
    print(pt.round(4).to_string(index=False))

    S = pd.concat(sub_rows)
    ss = S.groupby(["subtype", "family", "n"], sort=False)[sub_names].agg(["mean", "std"])
    ss.to_csv(f"{a.out}/exp5_subtype_recall_multiseed.csv")
    print("\nAll tables saved in:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
