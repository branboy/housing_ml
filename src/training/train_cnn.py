import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.cnn_model import ConditionCNN
from src.data.image_processing import HousingImageDataset


def train_cnn(df, image_dir, epochs=4, batch_size=32, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = HousingImageDataset(df, image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConditionCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            preds = model(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model