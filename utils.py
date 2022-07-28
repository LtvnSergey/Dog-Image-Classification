import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms as T
from torchvision import models
import numpy as np
from torch.optim import Adam
from time import time
from torchvision.models import ResNet50_Weights
import os


# CLASS FOR DATASET
class ImageDataset(Dataset):

    def __init__(self, labels, root, transforms=None, valid=False):

        # Select training or validation set
        self.labels = labels[labels['is_valid'] == valid]

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = None

        self.root = root

    def __getitem__(self, idx):

        # Load image
        img_path = os.path.join(self.root, self.labels.iloc[idx].path)

        image = Image.open(img_path)

        if len(np.array(image).shape):
            image = image.convert('RGB')

        # Apply transforms to image
        if self.transforms:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)

        # Target class
        target = torch.tensor(self.labels.iloc[idx]['class'],
                              dtype=torch.long)

        # Class breed
        breed = self.labels.iloc[idx].breed

        return {'image': image, 'target': target, 'breed': breed}

    def __len__(self):
        return len(self.labels)


# Model class
class Model_Resnet50():

    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V1, n_classes=10):

        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device available: '+self.device)

        # Create model
        self.model = models.resnet50(weights=weights)
        self.freeze_weights()
        self.add_output_layer(n_classes)
        self.model.to(self.device)

        # Set criterion
        self.criterion = nn.CrossEntropyLoss()

        # Set optimizer
        self.optimizer = Adam(params=self.model.parameters(), lr=0.001)


    # Freeze pretrained weights
    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False

    # Add layer with N classes
    def add_output_layer(self, n_classes):
        n_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(n_inputs, 256),
                                      nn.ReLU(),
                                      nn.Dropout(0.4),
                                      nn.Linear(256, n_classes))

    # Accuracy score
    def accuracy(self, output, targets):
        return np.count_nonzero((np.argmax(np.array(nn.Softmax(dim=1)(output.cpu()).detach()), axis=1) - np.array(targets.cpu())) == 0) / \
               targets.shape[0]

    # Train model or evaluate
    def train_eval_epoch(self, dataloader, mode, print_results=True):

        if mode =='train':
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_acc = 0.0

        start = time()

        with torch.set_grad_enabled(mode == 'train'):

            n_batches = 0
            for batch in dataloader:
                # Get images and targets from batch
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)

                if mode == 'train':
                    # Clear gradient
                    self.optimizer.zero_grad()

                # Forward pass
                output = self.model(images)

                # Loss
                loss = self.criterion(output, targets)

                if mode =='train':
                    # Backward propagation
                    loss.backward()

                    # Optimizer step
                    self.optimizer.step()

                # Collect train loss for further average estimation
                total_loss += loss.item()

                # Collect train accuracy for further average estimation
                total_acc += self.accuracy(output, targets)

                # Calculate batches
                n_batches += 1

        # Estimate average loss and accuracy
        total_loss = total_loss / n_batches
        total_acc = total_acc / len(dataloader)

        if print_results:
            print('| %s Loss:%.3f | %s Acc:%.3f | Time elapsed:%.2f s |' %
                  (mode, total_loss, mode, total_acc, time() - start))

        return total_loss, total_acc


    # Train or/and evaluate for epoches
    def train_eval(self, train_dl, valid_dl, n_epoches, print_results=True,
                   history_file='models/history/history.csv', save_file_name='models/history/best_state.cpkt'):

        history = []
        valid_loss_min = np.Inf
        train_loss, train_acc, valid_loss, valid_acc = None, None, None, None

        # Epoch train/eval cycle
        for epoch in range(n_epoches):

            if print_results:
                print(f'\n----- Epoch: {epoch+1} -----')

            if train_dl:
                # Train
                train_loss, train_acc = self.train_eval_epoch(train_dl, mode='train', print_results=True)

            if valid_dl:
                # Validate
                valid_loss, valid_acc = self.train_eval_epoch(valid_dl, mode='valid', print_results=True)

            # Save history
            if history_file:
                history.append([epoch, train_loss, valid_loss, train_acc, valid_acc])

        # Save history
        if history_file:
            history = pd.DataFrame(history, columns=['epoch',
                                                     'train_loss',
                                                     'valid_loss',
                                                     'train_acc',
                                                     'valid_acc'])
            history.to_csv(history_file, index=False)

        if (save_file_name != None) and (valid_dl != None):
            # Save best state
            valid_loss_min = np.Inf
            if valid_loss < valid_loss_min:
                torch.save(self.model.state_dict(), save_file_name)
                valid_loss_min = valid_loss
        pass

    # Make prediction
    def predict(self, x):

        # Disable gradients
        with torch.no_grad():
            self.model.eval()

            # Predict logits
            logits = self.model(torch.unsqueeze(x, dim=0))
            # Get probabilities from logits
            predict_proba = nn.Softmax()(logits)
            # Get class with highest probability
            predict = torch.argmax(nn.Softmax()(logits))

        return predict, predict_proba


# PRERPOCESS LABELS
def preprocess_labels(label_file):

    # Dictionary with dir names : breeds
    breeds_dict = {
        'n02093754': 'Australian terrier',
        'n02089973': 'Border terrier',
        'n02099601': 'Samoyed',
        'n02087394': 'Beagle',
        'n02105641': 'Shih-Tzu',
        'n02096294': 'English foxhound',
        'n02088364': 'Rhodesian ridgeback',
        'n02115641': 'Dingo',
        'n02111889': 'Golden retriever',
        'n02086240': 'Old English sheepdog'
    }

    # Dictionary with class number : breed
    class_dict = {label: i for i, label in enumerate(breeds_dict.keys())}

    # Load label file
    labels = pd.read_csv(label_file)
    labels['breed'] = labels['noisy_labels_0'].map(breeds_dict)
    labels['class'] = labels['noisy_labels_0'].map(class_dict)

    return labels[['path', 'class', 'breed', 'is_valid']]



# CREATE TRAIN AND VALIDATION DATALOADERS
def create_dataloader(labels, root, transforms, valid, shuffle, batch_size=64):

    # Create dataset
    dataset = ImageDataset(labels, root=root,
                           transforms=transforms,
                           valid=valid)

    # Create dataloader
    dl = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=shuffle)
    return dl


# Load image
def load_image(filename):

    # Image transforms
    transforms = T.Compose([T.Resize(size=(224, 224)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])
    # Open image
    image = Image.open(filename)

    # Make image RGB
    if len(np.array(image).shape):
        image = image.convert('RGB')

    image = transforms(image)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transfer image to device
    image = image.to(device)

    return image
