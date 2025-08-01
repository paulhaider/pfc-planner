import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

import utils

# --- ANN Module ---

class ANNPlanner(nn.Module):
    def __init__(self, num_choices, trajectory_length):
        super(ANNPlanner, self).__init__()
        self.num_choices = num_choices
        self.trajectory_length = trajectory_length
        # Simplified CNN for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # Calculate input features for the linear layers
        self._dummy_input_shape = (1, 3, 100, 100) # Assuming 100x100 images
        dummy_input = torch.rand(self._dummy_input_shape)
        conv_output_size = self.conv_layers(dummy_input).size(1)

        # Output for *entire trajectory* (regression of a vector)
        # This layer will output 'trajectory_length' number of values
        self.trajectory_regressor = nn.Linear(conv_output_size, trajectory_length)

        # Output for choice (left/right - classification)
        self.choice_classifier = nn.Linear(conv_output_size, num_choices)

    def forward(self, x):
        x = torch.unflatten(x, 1, (3, 100, 100))  # Ensure input is in correct shape
        features = self.conv_layers(x)
        predicted_trajectory = self.trajectory_regressor(features)
        choice_logits = self.choice_classifier(features)
        # in case of an ANN we would simply return
        # return predicted_trajectory, choice_logits
        # but to align with GLE we return a single tensor
        return torch.cat((predicted_trajectory, choice_logits), dim=1)

# --- Dataset and DataLoader ---

class RobotArmDataset(torch.utils.data.Dataset):
    def __init__(self, image_data, transform=None):
        self.image_data = image_data
        self.transform = transform
        self.choice_to_idx = {'left': 0, 'right': 1}

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        item = self.image_data[idx]
        image = Image.open(item['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Target for trajectory regression (the full sequence)
        target_trajectory = torch.tensor(item['ground_truth_trajectory'], dtype=torch.float)

        # Target for choice classification
        target_choice_idx = self.choice_to_idx[item['target_choice']]

        return image, target_trajectory, torch.tensor(target_choice_idx, dtype=torch.long)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting ANN Planner for Robotic Arm...")
    # Define your data directory relative to where you run this script
    DATA_DIR = './data/' # Make sure this path is correct
    print("Using data from:", DATA_DIR)

    loaded_data = utils.load_all_predefined_data_and_config(DATA_DIR)

    if loaded_data is None:
        print("Failed to load data, exiting.")
        sys.exit(1) # Exit if data loading failed

    # Access the loaded data from the dictionary
    FLEXION_TRAJECTORY_DATA = loaded_data['FLEXION_TRAJECTORY_DATA']
    EXTENSION_TRAJECTORY_DATA = loaded_data['EXTENSION_TRAJECTORY_DATA']
    TASK_MAPPING = loaded_data['TASK_MAPPING']
    TRAJECTORY_LEN = loaded_data['TRAJECTORY_LEN']
    INITIAL_ELBOW_ANGLE = loaded_data['INITIAL_ELBOW_ANGLE']

    # Pass loaded data explicitly to get_image_paths_and_labels
    all_image_data = utils.get_image_paths_and_labels(
        DATA_DIR,
        FLEXION_TRAJECTORY_DATA,
        EXTENSION_TRAJECTORY_DATA,
        TASK_MAPPING,
        INITIAL_ELBOW_ANGLE
    )

    print(f"Loaded {len(all_image_data)} distinct data samples for training.")
    if not all_image_data:
        print("No image data found. Please check DATA_DIR and filename patterns.")
        sys.exit(1)

    image_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to a vector
    ])

    train_dataset = RobotArmDataset(all_image_data, transform=image_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(all_image_data), shuffle=True) # Use full batch for this small dataset

    num_choices = 2
    # Pass trajectory_length to the model
    model = ANNPlanner(num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)

    # --- Loss Functions ---
    # MSELoss for the trajectory regression (comparing sequences)
    criterion_trajectory = nn.MSELoss()
    # CrossEntropyLoss for the choice classification
    criterion_choice = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500  # More epochs might be needed for sequence regression
    print("\nStarting offline training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, true_trajectory, target_choice_idx) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(images)
            predicted_trajectory = output[:, :TRAJECTORY_LEN]  # First part is the trajectory
            choice_logits = output[:, TRAJECTORY_LEN:]  # Second part is the choice logits

            loss_trajectory = criterion_trajectory(predicted_trajectory, true_trajectory)
            loss_choice = criterion_choice(choice_logits, target_choice_idx)

            total_loss = loss_trajectory + loss_choice
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # Print loss less frequently due to high epoch count
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}") # Increased precision

    print("\nTraining finished.")

    # --- Save the model ---
    MODEL_SAVE_PATH = './models/trained_ann_planner.pth' # Choose a meaningful file name
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    from evaluate import evaluate_model
    evaluate_model(model, train_loader, all_image_data)
