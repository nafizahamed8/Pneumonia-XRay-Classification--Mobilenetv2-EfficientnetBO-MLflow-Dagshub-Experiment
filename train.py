# This entire cell is saved as 'train.py'

# 1. --- Imports ---
import os
import torch
import argparse
import mlflow
import dagshub
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_model(model_name, num_classes=2):
    """Loads a pretrained model and replaces the final layer."""
    model = None
    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise Exception("Invalid model. Choose 'mobilenetv2' or 'efficientnetb0'")
    return model

def get_data_loaders(data_dir, batch_size, img_size):
    """Applies transforms and creates DataLoaders."""
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3), # Convert grayscale to 3-channel
            transforms.RandomHorizontalFlip(),           # Augmentation
            transforms.RandomRotation(10),               # Augmentation
            transforms.ToTensor(),                       # Convert to tensor and scale [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=2, pin_memory=True)
                   for x in ['train', 'val']}
    
    return dataloaders, image_datasets['train'].classes

def calculate_class_weights(data_dir):
    """Calculates class weights to handle the data imbalance."""
    count_normal = len(os.listdir(os.path.join(data_dir, 'train', 'NORMAL')))
    count_pneumonia = len(os.listdir(os.path.join(data_dir, 'train', 'PNEUMONIA')))
    total = count_normal + count_pneumonia
    
    weight_for_0 = (1 / count_normal) * (total / 2.0)
    weight_for_1 = (1 / count_pneumonia) * (total / 2.0)
    
    class_weights = torch.tensor([weight_for_0, weight_for_1])
    print(f"Class Weights: NORMAL(0): {weight_for_0:.2f}, PNEUMONIA(1): {weight_for_1:.2f}")
    return class_weights

# --- 3. Main Training Function ---

def main(args):
    """The main function that runs the entire pipeline."""
    
    
    dagshub.init(repo_owner=args.repo_owner, repo_name=args.repo_name, mlflow=True)
    print("DagsHub MLflow tracking initialized.")

    data_dir = "data/raw/chest_xray"
    dataloaders, class_names = get_data_loaders(data_dir, args.batch_size, args.img_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_model(args.model_name).to(device)
    
    class_weights = calculate_class_weights(data_dir).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    with mlflow.start_run():
        print(f"Starting MLflow run for model: {args.model_name}")
        
        mlflow.log_params(vars(args))
        mlflow.log_param("class_names", class_names)
        mlflow.log_param("device", device)

        # --- Training Loop --- (unchanged)
        for epoch in range(args.epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                print(f"Epoch {epoch+1}/{args.epochs} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_acc", epoch_acc, step=epoch)

        
        print("Training complete. Logging model artifact...")
        
        model_path = f"{args.model_name}.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Model saved to {model_path} and logged to MLflow.")

# --- 4. Argument Parsing (Script Entry Point) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pneumonia Classification Training Pipeline")
    
    parser.add_argument("--repo_owner", type=str, default=os.getenv("DAGSHUB_REPO_OWNER"), help="DagsHub repo owner (your username)")
    parser.add_argument("--repo_name", type=str, default=os.getenv("DAGSHUB_REPO_NAME"), help="DagsHub repo name")
    
    parser.add_argument("--model_name", type=str, default="mobilenetv2", help="Model to train (mobilenetv2 or efficientnetb0)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    
    args = parser.parse_args()
    
    
    if not args.repo_owner or not args.repo_name:
        raise ValueError("Error: DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set as environment variables.")
    
    main(args)

print("\n✅ train.py script updated successfully.")
