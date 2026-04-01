import pandas as pd
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from PIL import Image
from tqdm import tqdm

from src.models.cnn_model import load_model, get_transform


# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "data/processed/fusion_dataset.csv"
OUTPUT_PATH = "data/processed/image_features.csv"

BATCH_SIZE = 32


# -----------------------------
# LOAD IMAGES IN BATCH
# -----------------------------
def load_image_batch(image_paths, transform):
    images = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            images.append(img)
        except:
            images.append(None)

    # Filter out bad images
    valid_images = [img for img in images if img is not None]

    if len(valid_images) == 0:
        return None, []

    batch_tensor = torch.stack(valid_images)
    return batch_tensor, images


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = pd.read_csv(DATA_PATH)

    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MODEL + TRANSFORM
    model = load_model(device)
    transform = get_transform()

    all_features = []
    all_image_ids = []

    print("Extracting features with batching...")

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[i:i+BATCH_SIZE]

        image_paths = batch_df["image_path"].tolist()
        image_ids = batch_df["image_id"].tolist()

        batch_images = []

        valid_ids = []

        # Load images
        for path, img_id in zip(image_paths, image_ids):
            try:
                img = Image.open(path).convert("RGB")
                img = transform(img)
                batch_images.append(img)
                valid_ids.append(img_id)
            except:
                continue

        if len(batch_images) == 0:
            continue

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            features = model(batch_tensor)

        # Flatten features
        features = features.view(features.size(0), -1)
        features = features.cpu().numpy()

        all_features.extend(features)
        all_image_ids.extend(valid_ids)

    # Convert to DataFrame
    feature_df = pd.DataFrame(all_features)
    feature_df["image_id"] = all_image_ids

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved image features!")


if __name__ == "__main__":
    main()