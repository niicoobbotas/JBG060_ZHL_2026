import numpy as np
import pandas as pd

TARGETS = [0.20, 0.10, 0.05]  # highest threshold with FNR at or below each target
keys = ["adm2_pcode", "target_year", "target_month", "lead"]

rf = pd.read_csv("processing_data/floods/forecasts_rf.csv")[keys + ["flood", "p_rf"]]
xg = pd.read_csv("processing_data/floods/forecasts.csv")[keys + ["p_flood"]].rename(columns={"p_flood": "p_xgb"})
lr = pd.read_csv("processing_data/floods/forecasts_lr.csv")[keys + ["p_lr"]]
d = rf.merge(xg, on=keys).merge(lr, on=keys)

grid = np.round(np.arange(0.001, 1.0, 0.001), 3)


def points_at_fnr(y, p):
    pred = p[:, None] >= grid[None, :]
    pos = (y == 1)[:, None]
    tp = (pred & pos).sum(0)
    fp = (pred & ~pos).sum(0)
    fn = (~pred & pos).sum(0)
    tn = (~pred & ~pos).sum(0)
    fnr = fn / (fn + tp)
    out = []
    for target in TARGETS:
        i = int(np.where(fnr <= target)[0].max())
        out.append({"target FNR": f"{target:.0%}", "threshold": grid[i],
                    "accuracy": (tp[i] + tn[i]) / len(y),
                    "precision": tp[i] / (tp[i] + fp[i]),
                    "FPR": fp[i] / (fp[i] + tn[i]),
                    "FNR": fnr[i],
                    "false alarms per hit": round(fp[i] / tp[i], 2)})
    return out


models = [("Logistic Regression", "p_lr"), ("XGBoost", "p_xgb"), ("Random Forest", "p_rf")]
rows = []
for h, s in d.groupby("lead"):
    for name, col in models:
        for r in points_at_fnr(s["flood"].values, s[col].values):
            rows.append({"lead": f"{h} month(s)", "model": name, **r})
t = pd.DataFrame(rows)
for c in ["accuracy", "precision", "FPR", "FNR"]:
    t[c] = (t[c] * 100).round(1).astype(str) + "%"
print(t.sort_values(["target FNR", "lead"], ascending=[False, True]).to_string(index=False))