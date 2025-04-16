import sys, os, argparse, yaml, torch

# Set the root directory of the project by going two levels up from the current file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Check if the 'datasets' folder exists in the root directory to make sure the path is correct
if "datasets" not in os.listdir(PROJECT_ROOT):
    raise RuntimeError("PROJECT_ROOT is misaligned.")  # If not, stop the program with an error

# Add the project root to Python's path so we can import project files from anywhere in the code
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Import utility functions and training engine
from utils.dataloaders import get_train_val_loaders  # Loads training and validation datasets
from utils.train_helpers import build_model           # Builds the CLIP model and classifier
from utils.naming import build_filename               # Creates consistent filenames for saving models and logs
from engine.train_classifier_stage1 import FinetuneTrainerStage1  # Training logic for Stage 1

# Main function to run the training
def main(config_path):
    # Load the configuration from the YAML file provided as an argument
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set the experiment name; use "stage1" if not defined in config
    config["experiment"] = config.get("experiment", "stage1")

    # Generate a base name for output files using the config and training stage info
    base_name = build_filename(config, config["epochs"], stage="image", extension="", timestamped=False)

    # Define where to save the trained model and the log file
    config["save_path"] = os.path.join(config["save_dir"], base_name + ".pth")  # Model file
    config["log_path"] = os.path.join(config["log_dir"], base_name + ".log")    # Log file

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and validation data, and get number of classes
    train_loader, val_loader, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes  # Save the number of classes in the config for later use

    # Build the CLIP model and the image classifier
    clip_model, classifier = build_model(config, freeze_text=True)  # We freeze the text encoder

    # Create the trainer object using the model, data, config, and device
    trainer = FinetuneTrainerStage1(clip_model, classifier, train_loader, val_loader, config, device)

    # Start the training process
    trainer.train()


# Run this block only if the script is run directly (not imported)
if __name__ == "__main__":
    # Parse the command line arguments to get the path to the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    # Call the main function with the provided config path
    main(args.config)
