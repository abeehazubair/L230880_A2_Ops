"""Download the CURRENT champion model ONCE from the DagsHub registry into models/.
The Flask app then loads models/champion.keras locally - it never calls DagsHub per request.
    python src/fetch_model.py
"""
import json
import shutil
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv(".env")
NAME, ALIAS = "VisionGuard", "champion"
OUT = Path("models")

client = MlflowClient()
mv = client.get_model_version_by_alias(NAME, ALIAS)
OUT.mkdir(exist_ok=True)
# SavedModel format: the .keras file of a partly-frozen nested model cannot be reloaded in Keras 2.15
target = OUT / "champion"
if target.exists():
    shutil.rmtree(target)
local = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="model/data/model")
shutil.copytree(local, target)


meta = {"name": NAME, "alias": ALIAS, "version": int(mv.version), "run_id": mv.run_id,
        "run_name": mv.tags.get("run_name"), "threshold": float(mv.tags["threshold"]),
        "img_size": int(mv.tags["img_size"])}
(OUT / "champion.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
print(f"Saved {target} (SavedModel folder, {size_mb:.1f} MB)")
