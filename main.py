from dataloading import (
    DATASET_INFO,
    calculate_mean_std,
    create_data_loaders,
    make_validation_splits,
    print_dataset_info,
)

from datalogging import writeResultsCSV
from models import MODELS
from train import train, evaluate
from transforms import apply_transformations
from utils import export_onnx, export_metadata, handle_resume, load_json, load_weights, save_weights, set_seed

import os
import time
import torch

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = time.strftime("%Y%m%d_%H%M%S")                  # Ugly - Used for filenaming
    timestamp_pretty = time.strftime("%d %B %Y %H:%M:%S")       # Pretty - used for printing 26 June 2026 14:30:22

    config  = load_json("config.json")                        # All run settings now in a dict called 'runSettings'

    dataset_config      = config["dataset"]                     # Shortcut for "dataset" config.json
    dataloading_config  = config["dataloading"]                 # Shortcut for "dataloading" config.json
    transform_config    = config["transformations"]             # Shortcut for "transformations" config.json
    model_config        = config["model"]                       # Shortcut for "model" config.json
    train_config        = config["training"]                    # Shortcut for "training" config.json

    dataset_name = dataset_config["dataset"]                       # Extract chosen dataset name
    dataset_info = DATASET_INFO[dataset_name]

    if train_config["seed"] is not None:
        set_seed(train_config["seed"])

    (mean,std) = calculate_mean_std(dataset_info)   # Return mean, std

    train_dataset, test_dataset = apply_transformations(dataset_info, transform_config, (mean,std))          # Apply Tranformations and data augmentation

    # Get classes for evaluation phase
    if dataset_name == "SVHN":
        class_names = [str(i) for i in range(10)]
    else:
        class_names = train_dataset.classes

    train_dataset, val_dataset = make_validation_splits(train_dataset, dataset_config)
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, dataloading_config)

    model_name      = model_config["name"]
    model           = MODELS[model_name](model_config, dataset_info)
    total_params    = sum(p.numel() for p in model.parameters())

    print()
    print(f"--- {model_name} on {dataset_name}-Dataset @ {timestamp_pretty} ---")
    print(f"Total Parameters: {total_params:,}")

    weights_path = handle_resume(train_config.get("resume"), model_name, os.path.dirname(__file__))
    if weights_path:
        model = load_weights(model, weights_path)
        print(f"Resumed from: {weights_path}")

    print()

    run_dir = os.path.join(os.path.dirname(__file__), "results", f"{model_name}-{timestamp}")
    export_metadata(model, config, timestamp, run_dir)

    print("--- Training ---")
    train_results = train(device, model, train_dataloader, val_dataloader, train_config)
    writeResultsCSV(train_results, f"{model_name}-{dataset_name}-train", os.path.join(run_dir, "training"))
    print()

    print("--- Testing ---")
    test_results = evaluate(device, model, test_dataloader, class_names)
    writeResultsCSV(test_results, f"{model_name}-{dataset_name}-test", os.path.join(run_dir, "testing"))

    save_weights(model, run_dir)
    export_onnx(model, dataset_info, run_dir, device)

if __name__ == "__main__":
    main()
