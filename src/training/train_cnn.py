import pandas as pd
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import joblib
from sklearn.model_selection import train_test_split
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

    print("Applying PCA...")

    # Build raw feature DataFrame — one row per image
    feature_df = pd.DataFrame(all_features)          # columns: 0, 1, …, 2047
    feature_df["image_id"] = all_image_ids

    # Fit PCA on training split only (prevent test-set leakage)
    df_meta = pd.read_csv(DATA_PATH)
    df_full = feature_df.merge(df_meta[["image_id"]], on="image_id")

    train_ids, _ = train_test_split(df_full["image_id"], test_size=0.2, random_state=42)
    train_mask   = feature_df["image_id"].isin(train_ids)

    X_train_img = feature_df.loc[train_mask].drop(columns=["image_id"])

    pca = PCA(n_components=200, random_state=42)
    pca.fit(X_train_img)                             # fit on train only

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    joblib.dump(pca, "outputs/models/pca.pkl")

    # Transform ALL rows (train + test) and save with named columns.
    # Named columns ("pca_0"…"pca_199") ensure training and inference use the
    # same feature space — previously the raw 2048-dim features were saved here
    # but inference applied PCA first, causing a silent feature mismatch.
    X_all = feature_df.drop(columns=["image_id"])
    X_all_pca = pca.transform(X_all)

    PCA_COLS = [f"pca_{i}" for i in range(pca.n_components_)]
    pca_df = pd.DataFrame(X_all_pca, columns=PCA_COLS)
    pca_df["image_id"] = feature_df["image_id"].values

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pca_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {pca.n_components_}-component PCA image features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()