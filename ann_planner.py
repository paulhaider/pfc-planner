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
        features = self.conv_layers(x)
        predicted_trajectory = self.trajectory_regressor(features)
        choice_logits = self.choice_classifier(features)
        return predicted_trajectory, choice_logits

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

            predicted_trajectory, choice_logits = model(images)

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
    MODEL_SAVE_PATH = 'trained_ann_planner.pth' # Choose a meaningful file name
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    print("\n--- Demonstration of Inference ---")
    model.eval()
    with torch.no_grad():
        print("Evaluating on training data:")
        for i, (images, true_trajectory, true_choice_idx) in enumerate(train_loader):

            for k in range(len(all_image_data)):
                single_image = images[k].unsqueeze(0) # Get one image from batch and add batch dim
                single_true_trajectory = true_trajectory[k]
                single_true_choice_idx = true_choice_idx[k]

                # Get the corresponding original item from all_image_data
                original_item_data = all_image_data[k] # This is safe since batch_size is full dataset and shuffle is applied to `all_image_data` once.

                # Make predictions for the single image
                predicted_trajectory_tensor, pred_choice_logits = model(single_image)
                predicted_trajectory = predicted_trajectory_tensor.squeeze(0).cpu().numpy() # Remove batch dim, to numpy

                _, predicted_choice_idx = torch.max(pred_choice_logits, 1)
                predicted_choice = 'left' if predicted_choice_idx.item() == 0 else 'right'

                true_choice = 'left' if single_true_choice_idx.item() == 0 else 'right'

                print(f"\n--- Input Image: {os.path.basename(original_item_data['image_path'])} ---")
                print(f"Initial Angle (Hardcoded): {original_item_data['initial_angle']}°")
                print(f"Target Final Angle (from filename): {original_item_data['target_final_angle']}°")
                print(f"Calculated Angle Difference: {original_item_data['angle_difference']}°")
                print(f"True Choice: {true_choice}")
                print(f"Predicted Choice (Left/Right): {predicted_choice}")

                # Compare predicted and true trajectories
                print(f"True Trajectory (first 5 points): {single_true_trajectory.cpu().numpy()[:5]}")
                print(f"Predicted Trajectory (first 5 points): {predicted_trajectory[:5]}")

                # Optional: You could plot these trajectories to visualize the fit
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(utils.rad2deg(single_true_trajectory.cpu().numpy()), label='True Trajectory')
                plt.plot(utils.rad2deg(predicted_trajectory), label='Predicted Trajectory')
                plt.title(f"Trajectory for {os.path.basename(original_item_data['image_path'])}")
                plt.xlabel("Time Step")
                plt.ylabel("Elbow Angle (°)")
                plt.legend()
                plt.show()
