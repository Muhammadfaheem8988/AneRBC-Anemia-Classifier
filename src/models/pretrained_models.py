# Module: Pretrained CNN Architectures (Transfer Learning)

import torch.nn as nn
import torchvision.models as tv_models

def setup_pretrained_model(model_name: str, num_classes: int = 2):
    """
    Safely loads pretrained models, freezes the feature-extraction backbone,
    and replaces the classifier head for binary classification.
    """
    if model_name == 'mobilenet':
        # Load MobileNetV2
        model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'squeezenet':
        # Load SqueezeNet 1.1
        model = tv_models.squeezenet1_1(weights=tv_models.SqueezeNet1_1_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        
    elif model_name == 'resnet':
        # Load ResNet18
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError("Invalid model name. Choose 'mobilenet', 'squeezenet', or 'resnet'.")
        
    return model