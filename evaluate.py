#!/usr/bin/env python3
import os
import utils
from ann_planner import ANNPlanner, RobotArmDataset
from gle_planner import GLEPlanner
import torch
from torchvision import transforms

def evaluate_model(model, train_loader, all_image_data):
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
                for _ in range(20):
                    output = model(single_image)

                predicted_trajectory_tensor, pred_choice_logits = output[:, :len(single_true_trajectory)], output[:, len(single_true_trajectory):]
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
                plt.ylabel("Elbow Angle (rad)")
                plt.legend()
                plt.savefig(f"./results/{os.path.basename(original_item_data['image_path']).removesuffix('.bmp')}_trajectory.png")
                plt.show()
                plt.close()  # Close the plot to free memory

if __name__ == '__main__':
    print("Evaluating Planner models for Robotic Arm...")
    # Define your data directory relative to where you run this script
    EXPERIMENT_DIR = './' # Make sure this path is correct
    DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data/')
    print("Using data from:", DATA_DIR)

    # Load data by calling the function from the imported module
    # The function now returns the loaded data explicitly
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

    gle = GLEPlanner(tau=1.0, dt=0.01, num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)
    try:
        gle.load_state_dict(torch.load('./models/trained_gle_planner.pth'))
    except FileNotFoundError:
        print("GLE model file not found. Please ensure the model is trained and saved correctly.")
        sys.exit(1)
    ann = ANNPlanner(num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)
    try:
        ann.load_state_dict(torch.load('./models/trained_ann_planner.pth'))
    except FileNotFoundError:
        print("ANN model file not found. Please ensure the model is trained and saved correctly.")
        sys.exit(1)

    for model in [gle, ann]:
        print(f"Evaluating model: {model.__class__.__name__}")
        evaluate_model(model, train_loader, all_image_data)
        print(f"Finished evaluating model: {model.__class__.__name__}\n")
