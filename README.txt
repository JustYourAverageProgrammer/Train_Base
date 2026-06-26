--- TRAINBASE ---

Trainbase is one of my first hobby projects where I set out to create an easy to use training environment to experiment with CNN architecture and training.

I wanted to implement an easy way for users to specify and implement training variables and parameters in a simple to use JSON file.

This would help me quickly adjust training parameters and develop my own CNN architectures. Trainbase will hopefully come with my own CNN model "Argus" named after the 100 eyed giant, along with the ability to define your own within the models.py file.

This is also my first non-trivial Python project, so apologies in advance for any bad practices or non-pythonic syntax!

See below for a quick user guide...


--- CONFIGURATION ---

Everything in Trainbase was made to be controlled as best as possible through a single JSON config file. There is no need to touch the code to change training behaviour, just edit the JSON and run.

Any parameter set to null will fall back to a sensible default. For most parameters this means inheriting from the global training settings or PyTorch library defaults. For example, if lr is null inside an optimizer's params block, it will use the top-level learning_rate value. The same applies to weight_decay. This means you only need to specify overrides where you want behaviour that differs from the global settings.

Global training settings:

    "training": {
        "epochs":        200,
        "learning_rate": 1e-2,
        "weight_decay":  1e-4,
        "seed":          null       <- null means no fixed seed (non-reproducible runs)
    }

Criterion, scheduler, and optimizer each follow the same pattern: set "name" to select which one to use, then configure its parameters under "params". Only the selected one is used, the rest are ignored. Any null parameter falls back to either the PyTorch default or the global learning_rate / weight_decay / epochs where applicable.

For example, CosineAnnealingLR's T_max defaults to the global epochs value if left null, and OneCycleLR's max_lr defaults to the global learning_rate.


--- DATASETS ---

The dataset is configured through the JSON file under the "dataset" and "dataloading" blocks. Setting "dataset" to any supported name is all that is required — Trainbase will automatically handle the number of classes, input channels, and image size for that dataset. val_split controls what fraction of the training data is held out for validation.

Supported datasets out of the box:

    CIFAR10, CIFAR100           <- 32x32 RGB, 10/100 classes
    MNIST, FashionMNIST, KMNIST <- 28x28 grayscale, 10 classes
    SVHN                        <- 32x32 RGB, 10 classes (street house numbers)
    STL10                       <- 96x96 RGB, 10 classes

Argus can be swapped onto any dataset without code changes num_classes, num_channels, and image_size are all passed directly into the model constructor, so the architecture adapts automatically. Switching from CIFAR10 to MNIST or STL10 is just a one-line JSON change, Although I cannot guarantee performance as Argus was best designed for CIFAR10 at the time of writing.


--- RESUMING TRAINING ---

The "resume" field under "training" controls whether to start fresh or continue from a previous run's weights.pth file.

Each completed run saves a weights.pth file inside a timestamped folder under results/, named like:

    results/argus_v2-20260626_143022/weights.pth

The resume field accepts the following values:

    null            - fresh training run, no weights loaded
    "latest"        - loads weights from the most recent run for this model
    "oldest"        - loads weights from the oldest run for this model
    "20260626"      - loads the latest weights from the run matching this timestamp fragment
    "path/to/..."   - loads weights directly from an exact file path

Examples:

    "resume": null               start fresh
    "resume": "latest"           continue from most recent run
    "resume": "argus_v2"         matches any run folder containing "argus_v2"
    "resume": "20260626_143022"  matches a specific run by timestamp

Note: resume only loads the model weights. It does not restore the epoch count or optimizer state, so training will always start from epoch 0 with your currently configured number of epochs. This is intentional — resume is designed for continuing a fully completed run into further training, not for recovering a crashed mid-run. I may implement that feature down the line...


--- ONNX EXPORT ---

At the end of a training run, Trainbase will automatically export the trained model to ONNX format, producing two files in the run directory:

    model.onnx    <- the model graph and architecture
    model_onnx.data                     <- the external weight data (split out due to model size)

These are intended for use with visualisation tools and other deep learning utilities.

--- RESULTS ---

At the end of each run, Trainbase outputs two CSV files into subfolders of the run directory:

    training/{model_name}-{dataset_name}-train.csv  <- per-epoch training metrics
    testing/{model_name}-{dataset_name}-test.csv    <- per-class test accuracy

The training CSV logs Epoch, Train Loss, Val Loss, Train Accuracy, and Time (in seconds) for every epoch, useful for plotting learning curves and spotting overfitting. The testing CSV breaks down final model accuracy per class along with an overall Total accuracy, giving a quick view of where the model performs well and where it struggles.


--- ARGUS CNN ---

Argus is the CNN architecture developed alongside Trainbase, named after the hundred-eyed giant of Greek mythology. It was designed and iterated on entirely within Trainbase across many training runs, using the JSON config to rapidly experiment with depth, dropout, learning rate schedules, and optimizers.

The architecture consists of five convolutional blocks, each containing two Conv2d layers with BatchNorm and Leaky ReLU, progressively doubling channel depth from 32 up to 512. MaxPooling is applied between blocks to downsample spatially, and a Network-in-Network head (1x1 convolution + Global Average Pooling) is used in place of expensive fully connected layers to produce the final class logits. 2D-Dropout is applied before the NiN head to prevent over-reliance on individual feature maps.

After iterating on the architecture and training configuration through Trainbase, Argus was able to acheive 90%+ accuracy on CIFAR-10.