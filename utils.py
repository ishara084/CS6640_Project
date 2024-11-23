import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Image transformation pipeline in a standard format
def get_transform(resize_image_size):
    return transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),  # Convert grayscale-like images to 3 channels
        transforms.Resize(resize_image_size),  # Resize images
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
    ])


# Prepare test/train/val data
def prepare_train_test_data(transform):
    train_val_dataset = SARImageDataset(csv_file="data/train.csv", image_folder="data/images_train",transform=transform)
    test_dataset = SARImageDataset(csv_file="data/test.csv", image_folder="data/images_test", transform=transform)

    # Extract labels from the train_val_dataset for stratification
    labels = []
    for idx in range(len(train_val_dataset)):
        _, label = train_val_dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        labels.append(label)

    # Perform stratified split
    train_indices, val_indices = train_test_split(
        list(range(len(train_val_dataset))),
        test_size=0.2,  # 20% for validation
        stratify=labels,
        random_state=42  # For reproducibility
    )

    # Create training and validation subsets
    train_dataset = Subset(train_val_dataset, train_indices)
    val_dataset = Subset(train_val_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


# Data Handler Class for image preparations
class SARImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"{self.image_folder}/{self.data.iloc[idx, 0]}.png"
        label = self.data.iloc[idx, 1]
        image = Image.open(img_name)

        # Ensure image is in 'RGB' mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert label to appropriate data type
        label = int(label)
        return image, label


# Common method to calculate and print evaluation metrics
def calculate_evaluation_metrics(all_labels, all_preds, model_keyword):
    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["No Internal Waves", "Internal Waves"]))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Save prediction data to a CSV file (As a backup)
    results_df = pd.DataFrame({
        "true_labels": all_labels,
        "predicted_labels": all_preds
    })

    file_path = f"model_outputs_data/model_prediction_logs/{model_keyword}_labels_predictions.csv"
    results_df.to_csv(file_path, index=False)
    print(f"\n Predictions saved to {file_path}")