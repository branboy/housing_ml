import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


# -----------------------------
# LOAD MODEL (GPU SUPPORT)
# -----------------------------
def load_model(device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Remove classification head
    model = torch.nn.Sequential(*list(model.children())[:-1])

    model.to(device)
    model.eval()

    return model


# -----------------------------
# TRANSFORM
# -----------------------------
def get_transform():
    weights = ResNet50_Weights.DEFAULT
    return weights.transforms()


# -----------------------------
# EXTRACT FEATURES (ADD THIS BACK)
# -----------------------------
def extract_features(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image)

    return features.squeeze().cpu().numpy()