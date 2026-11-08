"""Task 2.6 - show the most confident misclassified TEST images of one MLflow run.
    python src/error_analysis.py --run-name mobilenetv2_lr0.001
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--run-name", default="mobilenetv2_lr0.001")
p.add_argument("--n", type=int, default=10)
args = p.parse_args()

load_dotenv(".env")
client = MlflowClient()
exp = client.get_experiment_by_name("VisionGuard")
run = client.search_runs([exp.experiment_id], f"attributes.run_name = '{args.run_name}'",
                         order_by=["attributes.start_time DESC"], max_results=1)[0]
thr = run.data.metrics["threshold"]
csv = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,
                                          artifact_path="evaluation/test_predictions.csv")

audit = pd.read_csv("reports/audit.csv")[["path", "width", "height", "size_bytes", "format", "mode"]]
df = pd.read_csv(csv).merge(audit, on="path")
wrong = df[df.label != df.pred].copy()
wrong["margin"] = (wrong.p_ai - thr).abs()            # how far on the wrong side of the threshold
fp = wrong[wrong.label == 0].nlargest(args.n // 2, "margin")          # real called AI
fn = wrong[wrong.label == 1].nlargest(args.n - len(fp), "margin")     # AI called real
picked = pd.concat([fp, fn]).reset_index(drop=True)
print(f"Run {args.run_name}: {len(wrong)} of {len(df)} test images wrong "
      f"({(wrong.label == 0).sum()} real->AI, {(wrong.label == 1).sum()} AI->real), threshold {thr:.2f}\n")

fig, axes = plt.subplots(2, 5, figsize=(20, 9.5))
for i, (ax, r) in enumerate(zip(axes.flat, picked.itertuples()), start=1):
    with Image.open(r.path) as im:
        ax.imshow(im.convert("RGB"))
    kind = "REAL called AI" if r.label == 0 else "AI called REAL"
    ax.set_title(f"#{i} {kind}\np(AI)={r.p_ai:.2f} | {r.width}x{r.height} {r.format} "
                 f"{r.size_bytes / 1024:.0f}KB {r.mode}", fontsize=10)
    ax.axis("off")
fig.suptitle(f"Most confident test mistakes - {args.run_name} (threshold {thr:.2f})", fontsize=14)
fig.tight_layout()
out = Path(f"reports/figures/fig_errors_{args.run_name}.png")
fig.savefig(out, dpi=110)

pd.set_option("display.width", 200)
print(picked[["label", "p_ai", "width", "height", "size_bytes", "format", "mode", "path"]]
      .assign(path=picked.path.str.split("/").str[-1].str[:45]).to_string())
print("\nSaved", out.as_posix())
