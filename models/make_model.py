import torch                                # PyTorch library for tensor operations and deep learning
import clip                                 # OpenAI's CLIP model library
import torch.nn as nn                       # Neural network layers and modules from PyTorch


def build_model(config, freeze_text=False):
    """
    Builds a CLIP-based model for classification.

    Args:
        config (dict): Configuration dictionary containing:
            - "model" (str): Either "vitb16" or "rn50" to select CLIP variant
            - "num_classes" (int): Number of output classes for the classification layer
        freeze_text (bool): If True, freezes CLIP's text encoder (transformer) during training

    Returns:
        clip_model (CLIP): The full CLIP model (image and text encoders)
        classifier (nn.Linear): A linear classifier head mapping image embeddings to class scores
    """

    # Automatically choose GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select CLIP model architecture based on config setting
    model_name = config["model"]
    clip_model, _ = clip.load("ViT-B/16" if model_name == "vitb16" else "RN50", device=device)

    # Optionally freeze text encoder (CLIP's transformer) to prevent updating its weights during training
    if freeze_text:
        for p in clip_model.transformer.parameters():
            p.requires_grad = False

    # Get the dimensionality of the image embedding output from CLIP (e.g., 512 or 1024)
    image_embed_dim = clip_model.visual.output_dim

    # Read the number of classification categories from config
    num_classes = config["num_classes"]

    # Define a linear classification layer to project image embeddings into class logits
    classifier = nn.Linear(image_embed_dim, num_classes)

    # Return the CLIP model and the classifier head separately
    return clip_model, classifier
