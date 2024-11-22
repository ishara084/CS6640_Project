import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Image transformation in a standard format
def get_transform(resize_image_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale-like images to 3 channels
        transforms.Resize(resize_image_size),  # Resize images
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
    ])


# Prepare test/train/val data
def prepare_train_test_data(transform):
    train_val_dataset = SARImageDataset(csv_file="data/train.csv", image_folder="data/images_train",
                                        transform=transform)
    test_dataset = SARImageDataset(csv_file="data/test.csv", image_folder="data/images_test", transform=transform)

    # Split into training and validation datasets
    train_size = int(0.8 * len(train_val_dataset))  # 80% for training
    val_size = len(train_val_dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    return [DataLoader(train_dataset, batch_size=32, shuffle=True),
            DataLoader(val_dataset, batch_size=32, shuffle=False),
            DataLoader(test_dataset, batch_size=32, shuffle=False)]


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

        if self.transform:
            image = self.transform(image)

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