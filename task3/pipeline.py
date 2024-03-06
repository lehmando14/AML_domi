import pickle
import gzip
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

################## Helper Functions ##################

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def resize_frame(frame, target_size):
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

def extract_labeled_frames(train_data, target_size=(128, 128)):
    labeled_frames = []
    labels = []

    for video_data in train_data:
        video = video_data['video']
        if 'label' not in video_data:
            label = np.zeros_like(video)
            frame_indices = range(video.shape[2])
        else:
            label = video_data['label']
            frame_indices = video_data['frames']

        for index in frame_indices:
            frame = video[:, :, index]
            frame_label = label[:, :, index]

            # Resize frame and label to the target size
            resized_frame = resize_frame(frame, target_size)
            resized_label = resize_frame(frame_label.astype(np.float32), target_size)

            # Convert boolean labels to integers
            resized_label = (resized_label > 0).astype(np.int64)

            labeled_frames.append(resized_frame)
            labels.append(resized_label)

    return np.array(labeled_frames), np.array(labels)

class MitralValveDataset(Dataset):
    def __init__(self, frames, labels, transform=None):
        self.frames = frames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]

        if self.transform:
            frame = self.transform(frame)

        # Reshape the frame and label to include the channel dimension
        frame = frame.reshape(1, 128, 128)  # For grayscale images
        label = label.reshape(1, 128, 128)  # Labels should match the frame shape

        frame = frame.float()
        label = torch.from_numpy(label).float() 

        return frame, label

############ Load the Data ############

train_data = load_zipped_pickle("train.pkl")
test_data = load_zipped_pickle("test.pkl")
samples = load_zipped_pickle("sample.pkl")

# Extract the labeled frames
labeled_frames, labels = extract_labeled_frames(train_data)


############ Create the Dataset and DataLoader ############

# Split the data into training and validation sets
# Adjust the test_size parameter as needed
train_frames, val_frames, train_labels, val_labels = train_test_split(
    labeled_frames, labels, test_size=0.2, random_state=42)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    # Add any other transformations here (e.g., normalization)
])

# Create the Dataset instances
train_dataset = MitralValveDataset(train_frames, train_labels, transform=transform)
val_dataset = MitralValveDataset(val_frames, val_labels, transform=transform)
# Create the DataLoader instances
batch_size = 8  # Adjust based on your GPU memory and training requirements

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

################## Define the Model ##################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Adjust the number of input channels for the convolution
            self.conv = DoubleConv(in_channels + (in_channels // 2), out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ToyUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ToyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits

import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=3, decoder_channels=(256, 128, 64), in_channels=1, classes=1)
# model = smp.MAnet(encoder_name='resnet34', encoder_depth=3, decoder_channels=(256, 128, 64), in_channels=1, classes=1)
# model = smp.PSPNet(encoder_name='resnet34', encoder_depth=3, in_channels=1, classes=1)
# model = ToyUNet(n_channels=1, n_classes=1)
    
################## Train the Model ##################
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(data.shape)
#     print(target.shape)
#     break

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Number of epochs
num_epochs = 100
best_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Iterate over data
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}', end='')

    # Validation phase
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        print(f', Validation Loss: {val_loss:.4f}')

        # Deep copy the model if it's the best so far
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

# Load best model weights
model.load_state_dict(best_model_wts)

# Save the best model
torch.save(model.state_dict(), 'best_model.pth')

################## Test the Model ##################

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = []
    
    for i, d in enumerate(test_data):
        labeled_frames, labels = extract_labeled_frames(test_data[i:i+1])
        test_dataset = MitralValveDataset(labeled_frames, labels, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # Make predictions 
        frame_predictions = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            prediction = model(inputs)[0][0]
            frame_predictions.append(prediction.cpu())
            # print(prediction.shape) 
            # print(prediction.max())
        frame_predictions=torch.stack(frame_predictions, -1)
        H, W = d['video'].shape[0:2]
        # print('H, W', H, W)
        # frame_predictions 128, 128, N_frame -> H, W, N_frame
        frame_predictions = frame_predictions.permute(2, 0, 1)
        frame_predictions = frame_predictions.unsqueeze(1)
        # print('frame_predictions', frame_predictions.shape)
        # Apply bilinear interpolation
        resized_tensor = F.interpolate(frame_predictions, size=(H, W), mode='bilinear', align_corners=False)
        # print('resized_tensor', resized_tensor.shape)
        # Remove the channel dimension to get back to (N_frame, H, W)
        prediction = resized_tensor.squeeze(1)
        prediction = prediction.permute(1, 2, 0)
        prediction=prediction>0.0
        predictions.append({
            'name': d['name'],
            'prediction': prediction.cpu().numpy()
            
        })   

# Save predictions
save_zipped_pickle(predictions, 'my_predictions.pkl')
