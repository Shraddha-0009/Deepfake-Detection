import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import random

# ------------------ PARAMETERS ------------------
BATCH_SIZE = 2  # videos per batch
NUM_CLASSES = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMG_SIZE = 380
SEQ_LEN = 16  # frames per video

train_dir = '/home/nvidia2/DeepFake/frames_dataset3/train'
val_dir = '/home/nvidia2/DeepFake/frames_dataset3/val'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ DATASET ------------------
class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, seq_len=16, transform=None):
        self.seq_len = seq_len
        self.transform = transform
        self.items = []  # list of (video_folder, label)
        classes = {'real': 0, 'fake': 1}
        for cls_name, label in classes.items():
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue
            for video_folder in sorted(os.listdir(class_dir)):
                video_path = os.path.join(class_dir, video_folder)
                if os.path.isdir(video_path):
                    self.items.append((video_path, label))

    def __len__(self):
        return len(self.items)

    def _load_frames(self, folder):
        frames = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
        if len(frames) == 0:
            raise ValueError(f"No frames in {folder}")
        if len(frames) < self.seq_len:
            # pad by repeating last frame
            repeats = (self.seq_len + len(frames) - 1) // len(frames)
            frames = (frames * repeats)[:self.seq_len]
        else:
            # uniform sampling
            indices = np.linspace(0, len(frames)-1, self.seq_len, dtype=int)
            frames = [frames[i] for i in indices]
        images = []
        for f in frames:
            img = Image.open(f).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        return torch.stack(images, dim=0)  # (seq_len, 3, H, W)

    def __getitem__(self, idx):
        folder, label = self.items[idx]
        frames = self._load_frames(folder)
        return frames, label


# ------------------ TRANSFORMS ------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = VideoFrameDataset(train_dir, seq_len=SEQ_LEN, transform=train_transform)
val_dataset = VideoFrameDataset(val_dir, seq_len=SEQ_LEN, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# ------------------ MODEL ------------------
class CNN_LSTM_Model(nn.Module):
    def __init__(self, lstm_hidden=256, num_classes=2, cnn_trainable=False):
        super().__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b4')
        self.cnn._fc = nn.Identity()  # remove classifier
        for param in self.cnn.parameters():
            param.requires_grad = cnn_trainable

        self.lstm = nn.LSTM(input_size=1792, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(lstm_hidden*2, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()  # batch, seq_len, channels, H, W
        x = x.view(B*T, C, H, W)
        features = self.cnn(x)  # (B*T, 1792)
        features = features.view(B, T, -1)  # (B, seq_len, 1792)
        lstm_out, _ = self.lstm(features)  # (B, seq_len, 2*lstm_hidden)
        lstm_out = lstm_out[:, -1, :]  # take last timestep
        x = self.fc1(lstm_out)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN_LSTM_Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ------------------ TRAIN/VALIDATION ------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)
    return running_loss/len(loader.dataset), running_corrects.double()/len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
    return running_loss/len(loader.dataset), running_corrects.double()/len(loader.dataset)


# ------------------ TRAIN LOOP ------------------
best_acc = 0.0
start_epoch = 0

# If resuming training, uncomment this:
# checkpoint = torch.load('cnn_lstm_checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch'] + 1
# best_acc = checkpoint['best_acc']

for epoch in range(start_epoch, NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    # --- Save checkpoint (always) ---
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, 'cnn_lstm_checkpoint.pth')

    # --- Save best model separately ---
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'cnn_lstm_deepfake_best_all.pth')
        print("Saved best model!")

print("Training complete.")
