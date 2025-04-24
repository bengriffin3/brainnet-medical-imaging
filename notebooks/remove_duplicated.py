import os
import hashlib
from pathlib import Path
from PIL import Image
import kagglehub
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

def compute_image_hash(image_path: str, hash_algorithm="md5") -> str:
    """
    Compute the hash of an image file.

    Args:
        image_path: Path to the image file.
        hash_algorithm: Hashing algorithm to use (default is 'md5').

    Returns:
        A hash string representing the image content.
    """
    hash_func = hashlib.new(hash_algorithm)
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure consistent format
        img_data = img.tobytes()  # Convert image to bytes
        hash_func.update(img_data)
    return hash_func.hexdigest()

def find_and_remove_duplicate_images(data_dir: str):
    """
    Find and remove duplicate images in a dataset directory.

    Args:
        data_dir: Path to the dataset directory.
    """
    image_hashes = {}
    duplicates = []

    # Ensure the base directory is the parent of 'Training' and 'Testing'
    base_dir = os.path.abspath(data_dir)

    # Traverse the dataset directory
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Compute the hash of the image
                image_hash = compute_image_hash(file_path)

                if image_hash in image_hashes:
                    # Duplicate found
                    original_file = image_hashes[image_hash]
                    # Store relative paths starting from the base directory
                    relative_duplicate = os.path.relpath(file_path, base_dir)
                    relative_original = os.path.relpath(original_file, base_dir)
                    duplicates.append((relative_duplicate, relative_original))
                else:
                    image_hashes[image_hash] = file_path
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Remove duplicates and log details
    for duplicate, original in duplicates:
        print(f"Duplicate found: {duplicate} (same as {original})")
        print(f"Removing duplicate: {duplicate}")
        os.remove(os.path.join(base_dir, duplicate))

    print(f"Total duplicates removed: {len(duplicates)}")

def setup_data(batch_size: int = 64) -> tuple:
    """
    Set up data loaders and datasets for training and validation.

    Args:
        batch_size: Number of samples per batch.

    Returns:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        train_set: Training dataset.
        val_set: Validation dataset.
    """
    # Download dataset
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    data_dir = pathlib.Path(path)

    # Remove duplicate images
    find_and_remove_duplicate_images(data_dir)

    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_set = torchvision.datasets.ImageFolder(
        data_dir.joinpath('Training'),
        transform=transform
    )
    val_set = torchvision.datasets.ImageFolder(
        data_dir.joinpath('Testing'),
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f'Training samples: {len(train_set)}')
    print(f'Validation samples: {len(val_set)}')
    print(f'Number of classes: {len(train_set.classes)}')
    print(f'Classes: {train_set.classes}')

    return train_loader, val_loader, train_set, val_set

# Set up data loaders
train_loader, val_loader, train_set, val_set = setup_data(batch_size=64)