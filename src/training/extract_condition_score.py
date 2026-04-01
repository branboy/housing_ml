import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_condition_scores(model, dataset, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    scores = []

    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(device)
            outputs = model(images)
            scores.append(outputs.cpu().numpy())

    return np.vstack(scores)