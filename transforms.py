import torchvision.datasets as datasets
import torchvision.transforms as transforms

def apply_transformations(dataset_info, transform_config, mean_std):

    # Unpack mean_std
    mean, std = mean_std

    # Transformation definition according to config.json NOTE: split into pre and post ToTEnsor()
    augmentations_preToTensor = []
    augmentations_postToTensor = []

    if transform_config["random_horizontal_flip"]["enabled"]:
        params = {k: v for k, v in transform_config["random_horizontal_flip"].items() if k != "enabled" and v is not None}
        augmentations_preToTensor.append(transforms.RandomHorizontalFlip(**params))

    if transform_config["random_crop"]["enabled"]:
        params = {k: v for k, v in transform_config["random_crop"].items() if k != "enabled" and v is not None}
        augmentations_preToTensor.append(transforms.RandomCrop(dataset_info.image_size, **params))

    if transform_config["color_jitter"]["enabled"]:
        params = {k: v for k, v in transform_config["color_jitter"].items() if k != "enabled" and v is not None}
        augmentations_preToTensor.append(transforms.ColorJitter(**params))

    if transform_config["random_rotation"]["enabled"]:
        params = {k: v for k, v in transform_config["random_rotation"].items() if k != "enabled" and v is not None}
        augmentations_preToTensor.append(transforms.RandomRotation(**params))

    if transform_config["random_erasing"]["enabled"]:
        params = {k: v for k, v in transform_config["random_erasing"].items() if k != "enabled" and v is not None}
        augmentations_postToTensor.append(transforms.RandomErasing(**params))

    train_transformlist = transforms.Compose([

        *augmentations_preToTensor,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        *augmentations_postToTensor

    ])

    test_transformlist = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize(mean, std)

    ])

    # Load Datasets
    if  dataset_info.torch_class == datasets.SVHN:           # NOTE: SVHN uses split='train' as opposed to train = true/false
        train_dataset = dataset_info.torch_class(root = './datasets', split = 'train', transform = train_transformlist, download = True)
        test_dataset  = dataset_info.torch_class(root = './datasets', split = 'test', transform = test_transformlist, download = True)

        return train_dataset, test_dataset

    else:
        train_dataset = dataset_info.torch_class(root = './datasets', train = True,  transform = train_transformlist, download = True)
        test_dataset  = dataset_info.torch_class(root = './datasets', train = False, transform = test_transformlist, download = True)

        return train_dataset, test_dataset
