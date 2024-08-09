# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:23:36 2024

@author: Alfred
"""

import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

#%% Read GDF file
# Load the data
file_path = './BCICIV_2a_gdf/A02T.gdf'
#file_path = './BCICIV_2a_gdf/A02E.gdf'

# Set the channel names for EEG and EOG
ch_names = ['Fz', 'FC3','FC1','FCz','FC2','FC4', 'C5','C3','C1','Cz','C2','C4','C6',
           'CP3','CP1','CPz','CP2','CP4', 'P1','Pz','P2', 'POz']
eog_names = ['EOG-left', 'EOG-central', 'EOG-right']

raw = mne.io.read_raw_gdf(file_path, preload=True, eog=eog_names)

mapping = {'EEG-Fz':'Fz', # names for EEG channels
           'EEG-0':'FC3',
           'EEG-1':'FC1',
           'EEG-2':'FCz',
           'EEG-3':'FC2',
           'EEG-4':'FC4',
           'EEG-5':'C5',
           'EEG-C3':'C3',
           'EEG-6':'C1',
           'EEG-Cz':'Cz',
           'EEG-7':'C2',
           'EEG-C4':'C4',
           'EEG-8':'C6',
           'EEG-9':'CP3',
           'EEG-10':'CP1',
           'EEG-11':'CPz',
           'EEG-12':'CP2',
           'EEG-13':'CP4',
           'EEG-14':'P1',
           'EEG-Pz':'Pz',
           'EEG-15':'P2',
           'EEG-16':'POz',
           }

mne.rename_channels(raw.info, mapping) # Remapping electrodes position

'''
# #%% ICA - Future work?
# montage = mne.channels.make_standard_montage('standard_1020')
# raw.set_montage(montage)

# #%%
# from mne.preprocessing import create_eog_epochs
# eog_evoked = create_eog_epochs(raw).average()
# eog_evoked.apply_baseline(baseline=(None, -0.2))
# eog_evoked.plot_joint()

# #%%
# from mne.preprocessing import ICA
# filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
# ica = ICA(n_components=22, max_iter='auto', random_state=97)
# ica.fit(filt_raw)
# ica

# #%%
# raw.load_data()
# ica.plot_sources(raw, show_scrollbars=False)
# #%%
# ica.plot_components()
'''

#%% Print Information of Data
raw.info

#%% Pick only EEG channels for analysis
eeg = raw.copy().pick(ch_names)

#%% Pre-processing
'''
The Dataset is already 50Hz notched, 1~100 BP.
'''
# Filter the data (1-40 Hz bandpass)
# raw.filter(1., 40., fir_design='firwin')

#%% Define events
events, event_id = mne.events_from_annotations(eeg)

#%% Epoching the data
tmin, tmax = 1,4   # Define time range for each epoch (0 to 4 seconds)
epochs = mne.Epochs(eeg, events, event_id={'769': 7, '770': 8, '771': 9, '772': 10}, tmin=tmin, tmax=tmax, baseline=None, preload=True)
# epochs = mne.Epochs(raw, events, event_id={'783': 7 }, tmin=tmin, tmax=tmax, baseline=None, preload=True)
# Extract data and labels
X = epochs.get_data(copy=True)  # shape (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]-7  # Adjust labels to start from 0

#%% Preparing Data
# Normalize data
X = (X - np.mean(X, axis=2, keepdims=True)) / np.std(X, axis=2, keepdims=True)

#%% Split into Train/Validation Dataset

# Creating a PyTorch dataset and dataloader
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and dataloader
dataset = EEGDataset(X, y)

validation_split = .2
shuffle_dataset = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=16, sampler=valid_sampler)

#%%

# Defining EEGNet in PyTorch
class EEGNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=751, dropoutRate=0.5,
                 kernLength=250, F1=8, D=2, F2=16, dropoutType='Dropout'):
        
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.num_classes = nb_classes
        self.kL = kernLength
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kL), padding='same', bias=False),
            nn.BatchNorm2d(self.F1),
            nn.Conv2d(self.F1, self.F1 * self.D, (Chans, 1), groups=self.F1, padding=0),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding='same', groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2*(Samples // 32), nb_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

#%%
# Training the EEGNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, labels in train_loader:
        data, labels = data.unsqueeze(1).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    with torch.no_grad():
        for data, labels in valid_loader:
            data, labels = data.unsqueeze(1).to(device), labels.to(device)
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

print("Training complete.")
