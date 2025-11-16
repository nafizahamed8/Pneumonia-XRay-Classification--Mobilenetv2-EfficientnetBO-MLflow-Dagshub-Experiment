
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size, img_size):
    """Applies transforms and creates DataLoaders."""
    
    # Define a dictionary of transforms
    data_transforms = {
        # Train transform: includes data augmentation
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3), # Convert grayscale to 3-channel
            transforms.RandomHorizontalFlip(),           # Augmentation
            transforms.RandomRotation(10),               # Augmentation
            transforms.ToTensor(),                       # Convert to tensor and scale [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ]),
        # Validation transform: no augmentation
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # Create PyTorch ImageFolder Datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    # Create PyTorch DataLoaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=2, pin_memory=True)
                   for x in ['train', 'val']}
    
    # Return the loaders and the class names (['NORMAL', 'PNEUMONIA'])
    return dataloaders, image_datasets['train'].classes

def calculate_class_weights(data_dir):
    """Calculates class weights to handle the data imbalance."""
    # Count the number of files in each training class folder
    count_normal = len(os.listdir(os.path.join(data_dir, 'train', 'NORMAL')))
    count_pneumonia = len(os.listdir(os.path.join(data_dir, 'train', 'PNEUMONIA')))
    total = count_normal + count_pneumonia
    
    # Calculate weights. The rare class (NORMAL) will get a high weight.
    weight_for_0 = (1 / count_normal) * (total / 2.0)
    weight_for_1 = (1 / count_pneumonia) * (total / 2.0)
    
    class_weights = torch.tensor([weight_for_0, weight_for_1])
    print(f"Class Weights: NORMAL(0): {weight_for_0:.2f}, PNEUMONIA(1): {weight_for_1:.2f}")
    return class_weights

print("\n✅ preprocess.py script created successfully.")
