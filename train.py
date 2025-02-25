import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from wnet import WNet

class NucleusDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, ordinal_labels):
        self.images = images
        self.masks = masks
        self.ordinal_labels = ordinal_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        ordinal_label = self.ordinal_labels[idx]
        return image, mask, ordinal_label

def bce_loss(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)

def mse_loss(pred, target):
    return nn.MSELoss()(pred, target)

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks, ordinal_labels in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        ordinal_labels = ordinal_labels.to(device)

        optimizer.zero_grad()
        foreground_pred, ordinal_pred = model(images)


        masks = masks.float()
        ordinal_labels = ordinal_labels.float()

        # Calculate losses
        loss_bce = bce_loss(foreground_pred, masks)
        loss_mse = mse_loss(ordinal_pred, ordinal_labels)
        loss = loss_bce + 0.1 * loss_mse 

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WNet(num_grades=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # images =
    # masks =
    # ordinal_labels =
    dataset = NucleusDataset(images, masks, ordinal_labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(10):
        loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

if __name__ == "__main__":
    main()