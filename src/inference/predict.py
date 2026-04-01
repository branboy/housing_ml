import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import joblib

from src.models.cnn_model import ConditionCNN


# Load models ONCE
fusion_model = joblib.load("outputs/models/fusion_model.pkl")
scaler = joblib.load("outputs/models/scaler.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"

cnn_model = ConditionCNN()
state_dict = torch.load(
    "outputs/models/cnn_model.pth",
    map_location=device,
    weights_only=True
)
cnn_model.load_state_dict(state_dict)
cnn_model.to(device)
cnn_model.eval()


# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_condition_score(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        score = cnn_model(image)

    return score.cpu().numpy()[0][0]


def preprocess_structured_input(user_input, df_columns):
    """
    user_input = dict from UI
    df_columns = training columns (after encoding)
    """

    # Create empty vector
    x = np.zeros(len(df_columns))

    for i, col in enumerate(df_columns):
        if col in user_input:
            x[i] = user_input[col]

    return x


def predict_price(user_input, image, df_columns):
    # Step 1: condition score
    condition_score = get_condition_score(image)

    # Step 2: structured features
    structured_vector = preprocess_structured_input(user_input, df_columns)

    # Step 3: combine
    X = np.hstack([structured_vector, condition_score]).reshape(1, -1)

    # Step 4: scale
    X = scaler.transform(X)

    # Step 5: predict (log price)
    log_price = fusion_model.predict(X)[0]

    # Convert back
    price = np.exp(log_price)

    return price