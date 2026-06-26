import time
import torch

from torch import nn

import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


CRITERIONS = {
        "CrossEntropyLoss"  : nn.CrossEntropyLoss,        # Only real sensible option for the image classification tasks
}

OPTIMIZERS = {

        "Adam"      : optim.Adam,
        "AdamW"     : optim.AdamW,
        "SGD"       : optim.SGD,
        "rmsprop"   : optim.RMSprop,

}

SCHEDULERS = {

        "StepLR"                : scheduler.StepLR,
        "CosineAnnealingLR"     : scheduler.CosineAnnealingLR,
        "ReduceLROnPlateau"     : scheduler.ReduceLROnPlateau,
        "OneCycleLR"            : scheduler.OneCycleLR,
        "ExponentialLR"         : scheduler.ExponentialLR,

}

PLATEAU_SCHEDULERS = {"ReduceLROnPlateau"}
BATCH_SCHEDULERS   = {"OneCycleLR"}

# Helper function to resolve for global/exception paramaters
def get_params(k, v, train_config):

    if v is not None:
        return v

    if k == "learning_rate":
        return train_config["learning_rate"]

    if k == "weight_decay":
        return train_config["weight_decay"]

    if k in ("T_max", "epochs"):
        return train_config["epochs"]
    return None

global_params = {"weight_decay", "learning_rate", "T_max", "epochs"}

def train(device, model, train_loader, test_loader, train_config):

    model     = model.to(device)        # Send model to decive

    criterion_name   = train_config["criterion"]["name"]
    criterion_params = {k: v for k, v in train_config["criterion"]["params"][criterion_name].items() if v is not None}
    criterion        = CRITERIONS[criterion_name](**criterion_params)

    # --- READ OPTIMIZER CONFIG ---
    optimizer_name      = train_config["optimizer"]["name"]
    optimizer_params    = {
            k: get_params(k, v, train_config)
            for k, v in train_config["optimizer"]["params"][optimizer_name].items()
            if v is not None or k in global_params
            }
    optimizer           = OPTIMIZERS[optimizer_name](model.parameters(), **optimizer_params)

    # --- READ SHECDULER CONFIG ---
    scheduler_name      = train_config["scheduler"]["name"]
    scheduler_params = {
            k: get_params(k, v, train_config)
            for k, v in train_config["scheduler"]["params"][scheduler_name].items()
            if v is not None or k in global_params
    }
    scheduler           = SCHEDULERS[scheduler_name](optimizer, **scheduler_params)

    results = []    # List of Dicts to store training results

    for epoch in range(train_config["epochs"]):

        epoch_start = time.time() # Start time

        # --- Training phase ---

        model.train()       # Set model to training mode (enables dropout and batch norm training behaviour)
        train_loss = 0.0     # Accumulator for training loss this epoch

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)   # Move batch to GPU
            optimizer.zero_grad()                                   # Clear gradients from previous batch
            outputs = model(images)                                 # Forward pass
            loss    = criterion(outputs, labels)                    # Compute loss
            loss.backward()                                         # Backpropagate gradients
            optimizer.step()                                        # Update weights
            train_loss += loss.item()                                # Accumulate batch loss

            if scheduler_name in BATCH_SCHEDULERS:
                scheduler.step()

        # --- Evaluation phase ---

        model.eval()        # Set model to eval mode (disables dropout, freezes batch norm)
        correct  = 0        # Count of correct predictions
        total    = 0        # Count of total predictions
        test_loss = 0.0      # Accumulator for test loss this epoch

        with torch.no_grad():                                       # Disable gradient tracking for speed
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs   = model(images)                           # Forward pass
                test_loss += criterion(outputs, labels).item()       # Accumulate test loss
                predicted = outputs.argmax(dim=1)                   # Take highest scoring class as prediction
                correct  += (predicted == labels).sum().item()      # Count correct predictions
                total    += labels.size(0)                          # Count total predictions

        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total                            # Compute accuracy percentage

        if scheduler_name in PLATEAU_SCHEDULERS:
            scheduler.step(avg_test_loss)       # ReduceLROnPlateau needs a metric
        elif scheduler_name not in BATCH_SCHEDULERS:
            scheduler.step()                    # StepLR, CosineAnnealingLR, ExponentialLR

        epoch_time = time.time() - epoch_start      # Capture time of epoch

        print(f"Epoch {epoch+1:3d}/{train_config['epochs']} - Train Loss: {train_loss/len(train_loader):.3f} | Val Loss: {test_loss/len(test_loader):.3f} | Accuracy: {accuracy:.2f}% | Time: {epoch_time:.3f}s")

        results.append({
            "Epoch": epoch + 1,
            "Train Loss": train_loss / len(train_loader),
            "Val Loss": test_loss / len(test_loader),
            "Test Accuracy": accuracy,
            "Time": epoch_time,
        })

    return results

def evaluate(device, model, test_loader, class_names):

    model.eval()        # Set model to eval mode (disables dropout, freezes batch norm)

    class_correct = {}         # Count of correct predictions
    class_total   = {}         # Count of total predictions

    with torch.no_grad():                                           # Disable gradient tracking for speed

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)   # Move batch to GPU
            predicted = model(images).argmax(dim=1)                 # Forward pass and take highest scoring class

            for label,pred in zip(labels,predicted):                # Loop througheach item in batch
                label = label.item()                                # Converts tensor to plain python int
                class_total[label] = class_total.get(label, 0) + 1  # Add total seen count

                if label == pred.item():                            # If correct add to correct count for class
                    class_correct[label] = class_correct.get(label,0) + 1

    total_correct = sum(class_correct.values())
    total         = sum(class_total.values())
    overall_acc   = 100 * total_correct / total

    class_results = [
        {"Class": class_names[cls], "Accuracy": 100 * class_correct.get(cls, 0) / class_total[cls]}
        for cls in sorted(class_total)
    ]

    class_results.append({"Class": "Total", "Accuracy": overall_acc})

    print(f"Total Accuracy: {overall_acc:.3f}%")

    return class_results
