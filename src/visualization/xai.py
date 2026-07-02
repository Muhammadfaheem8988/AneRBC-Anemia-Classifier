# Name: Muhammad Faheem
# SRN: [Insert Your SRN Here]
# Module: Explainable AI (Integrated Gradients)

import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

def denormalize_image(tensor):
    """Reverses the ImageNet normalization so the image displays correctly."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def apply_integrated_gradients(model, input_tensor, target_class_idx, class_names, model_name):
    """
    Generates and plots an Integrated Gradients attribution map to explain 
    which pixels drove the model's classification decision.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Captum requires requires_grad=True on the input tensor
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Calculate attributions
    attributions, delta = ig.attribute(
        inputs=input_tensor, 
        target=target_class_idx, 
        return_convergence_delta=True
    )

    # Format tensors for visualization
    img_to_show = denormalize_image(input_tensor.squeeze(0).detach())
    attr_to_show = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    # Plot original image vs blended heat map
    fig, ax = viz.visualize_image_attr_multiple(
        attr_to_show,
        img_to_show,
        methods=["original_image", "blended_heat_map"],
        signs=["all", "all"],
        show_colorbar=True,
        titles=[f"Original Image", f"IG Attribution ({model_name})"],
        fig_size=(10, 5)
    )
    plt.tight_layout()
    plt.show()