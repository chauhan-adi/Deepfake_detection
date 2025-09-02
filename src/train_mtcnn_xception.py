import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch import optim
import timm  

# Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_videos_per_class=50):
        face_folder = os.path.join(root_dir, "numpy_faces_mtcnn")
        self.samples = []
        self.video_to_faces = {}
        for label_name, label in [("real",0), ("fake",1)]:
            class_dir = os.path.join(face_folder, label_name)
            video_count = 0
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy") and video_count < max_videos_per_class:
                    path = os.path.join(class_dir, fname)
                    video_name = fname.split("_")[0]
                    self.video_to_faces[video_name] = path
                    self.samples.append((path, label))
                    video_count += 1
        blink_path = os.path.join(root_dir, "blink_features.npy")
        self.blink_features = np.load(blink_path, allow_pickle=True).item() if os.path.exists(blink_path) else {}
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299,299)) 
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        img = np.load(npy_path)
        if self.transform:
            img = self.transform(img)
        video_name = os.path.basename(npy_path).split("_")[0]
        blink_feat = torch.tensor(self.blink_features.get(video_name, [0.0,0.0,0.0,0.0]), dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return img, blink_feat, label


# Model

class DeepfakeBlinkXception(nn.Module):
    def __init__(self, num_blink_features=4):
        super().__init__()
        
        self.base_model = timm.create_model('xception', pretrained=True, num_classes=0)
        self.fc = nn.Sequential(
            nn.Linear(self.base_model.num_features + num_blink_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, blink):
        x = self.base_model(x)
        x = torch.cat([x, blink], dim=1)
        x = self.fc(x)
        return x


# Training

def main():
    root_dir = "data/processed"
    dataset = DeepfakeDataset(root_dir, max_videos_per_class=10)  # small demo

    val_size = int(0.2*len(dataset))
    train_size = len(dataset)-val_size
    train_dataset, val_dataset = random_split(dataset, [train_size,val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DeepfakeBlinkXception().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):
        model.train()
        running_loss = 0
        for imgs, blink, labels in train_loader:
            imgs, blink, labels = imgs.to(device), blink.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, blink)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/3, Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        correct, total = 0,0
        with torch.no_grad():
            for imgs, blink, labels in val_loader:
                imgs, blink, labels = imgs.to(device), blink.to(device), labels.to(device)
                outputs = model(imgs, blink)
                preds = outputs.argmax(dim=1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)
        print(f"Validation Accuracy: {correct/total*100:.2f}%")

    print("Demo training with XceptionNet complete!")

if __name__=="__main__":
    main()
