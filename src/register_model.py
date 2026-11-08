"""Task 3.6 - register two VisionGuard runs as model versions; alias the best one 'champion'.
Run ONCE:  python src/register_model.py
"""
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv(".env")
NAME = "VisionGuard"
RUNS = [("mobilenetv2_lr0.001", "MobileNetV2, last block fine-tuned (reference run)"),
        ("exp_mnv2_unfreeze60", "MobileNetV2, 60 layers fine-tuned - best validation AUC (champion)")]
CHAMPION = "exp_mnv2_unfreeze60"

client = MlflowClient()
exp = client.get_experiment_by_name("VisionGuard")
versions = {}
for run_name, description in RUNS:
    run = client.search_runs([exp.experiment_id], f"attributes.run_name = '{run_name}'",
                             order_by=["attributes.start_time DESC"], max_results=1)[0]
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", NAME)
    client.update_model_version(NAME, mv.version, description=description)
    for key, value in {"run_name": run_name,
                       "threshold": run.data.metrics["threshold"],
                       "val_roc_auc": round(run.data.metrics["final_val_roc_auc"], 4),
                       "img_size": run.data.params["img_size"]}.items():
        client.set_model_version_tag(NAME, mv.version, key, str(value))
    versions[run_name] = mv.version
    print(f"Registered {NAME} version {mv.version}  <-  run {run_name} ({run.info.run_id[:8]})")

client.set_registered_model_alias(NAME, "champion", versions[CHAMPION])
print(f"Alias 'champion' -> {NAME} version {versions[CHAMPION]}")
