import json
import os
import random
import torch

def export_onnx(model, dataset_info, run_dir, device):
    model.eval()
    _input = torch.randn(1, dataset_info.num_channels, *dataset_info.image_size).to(device)
    torch.onnx.export(model, _input, os.path.join(run_dir, "model.onnx"))
    print(f"Model exported to {run_dir}/model.onnx")

def export_metadata(model, config, timestamp, run_dir):

    def strip_nulls(d):
        if isinstance(d, dict):
            return {k: strip_nulls(v) for k, v in d.items() if v is not None}
        return d

    optimizer_name = config["training"]["optimizer"]["name"]
    scheduler_name = config["training"]["scheduler"]["name"]

    # Build clean config - word for word but without nulls and only active optimizer/scheduler
    clean_config = strip_nulls({
        **config,
        "training": {
            **strip_nulls({k: v for k, v in config["training"].items() if k not in ("optimizer", "scheduler")}),
            "optimizer": {"name": optimizer_name, **strip_nulls(config["training"]["optimizer"]["params"][optimizer_name])},
            "scheduler": {"name": scheduler_name, **strip_nulls(config["training"]["scheduler"]["params"][scheduler_name])},
        }
    })

    metadata = {
        "timestamp"  : timestamp,
        "model_name" : config["model"]["name"],
        "architecture": model.__class__.__name__,
        "num_params" : sum(p.numel() for p in model.parameters()),
        **clean_config
    }

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

def handle_resume(resume, model_name, base_dir):
    if resume is None:                                                              # No resume, fresh training
        return None

    if os.path.exists(resume):                                                      # Exact path provided, use it directly
        return resume

    results_dir = os.path.join(base_dir, "results")                                 # Build path to results folder
    runs = [d for d in os.listdir(results_dir) if d.startswith(model_name)]         # Find all runs for this model

    if not runs:
        raise FileNotFoundError(f"No saved runs found for model '{model_name}'")    # No runs found for this model

    runs_sorted = sorted(runs, key=lambda d: d.split("-")[-1])                      # Sort by timestamp (last part after "-")
                                                                                    # split("-")[-1] grabs "20260626_143022" from "argus_v2-20260626_143022"
                                                                                    # sorted alphabetically which works since timestamps are YYYYMMDD_HHMMSS
    if resume == "latest":
        run = runs_sorted[-1]                                                       # Last item = most recent timestamp
    elif resume == "oldest":
        run = runs_sorted[0]                                                        # First item = oldest timestamp

    else:                                                                           # User provided a timestamp fragment e.g. "20260626"
        matches = [r for r in runs if resume in r]                                  # Check if their input appears anywhere in each folder name

        if not matches:
            raise FileNotFoundError(f"No run found matching timestamp '{resume}'")  # No runs match their timestamp
        run = matches[-1]                                                           # Take first match (should only be one if timestamp is specific enough, otherwise most recent)

    weights_path = os.path.join(results_dir, run, "weights.pth")                    # Build full path to weights file
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Run found but no weights.pth in '{run}'")         # Run folder exists but weights were never saved

    return weights_path


def load_json(path):              # Pass path in quotations "path/to/file.json"

    with open(path, "r") as file:
        json_file = json.load(file)

    return json_file


def load_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))

    return model


def save_weights(model, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "weights.pth"))
    print(f"Weights saved to {run_dir}/weights.pth")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
