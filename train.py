# Torch related libraries
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms

# Utilities
import random
import json
import os
import boto3
from PIL import Image
from io import BytesIO
import argparse
import logging
import sys
from datetime import datetime
from uuid import uuid4
import time

# Setting up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# ----------------------------------- Models ----------------------------------- #

class DNN_HEAD(nn.Module):
    def __init__(self, n_classes, ln_reg=None):
        super().__init__()

        # Initialize resnet
        self.resnet34 = models.resnet34()
        in_features = self.resnet34.fc.in_features
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-1])
        self.ln_reg = ln_reg

        # Add the Flatten layer
        self.flatten = nn.Flatten()

        # Create classfication head
        self.Linear = nn.Linear(in_features, n_classes)

        # Add Linear Regression Layer if used
        if self.ln_reg:
            self.Reg = nn.Linear(n_classes, 1)

    def freeze_backbone(self):
        for param in self.resnet34.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet34(x)
        x = self.flatten(x)
        x = self.Linear(x)
        if self.ln_reg:
            x = self.Reg(x)
        return x


class CNN_HEAD(nn.Module):
    def __init__(self, n_classes, ln_reg=None):
        super().__init__()

        # Initialize resnet
        self.resnet34 = models.resnet34()
        in_features = self.resnet34.fc.in_features
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-2])
        self.ln_reg = ln_reg

        # Add a convolutional layer to learn specific features
        # Convolutional layer
        self.conv = nn.Conv2d(in_features, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)  # ReLU activation

        # Add Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Add the Flatten layer
        self.flatten = nn.Flatten()

        # Create classfication head
        self.Linear = nn.Linear(256, n_classes)

        # Add Linear Regression Layer if used
        if self.ln_reg:
            self.Reg = nn.Linear(n_classes, 1)

    def freeze_backbone(self):
        for param in self.resnet34.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet34(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.Linear(x)
        if self.ln_reg:
            x = self.Reg(x)
        return x


class ATT_HEAD(nn.Module):
    def __init__(self, n_classes, embed_dim=512, num_heads=8, ln_reg=None):
        super().__init__()

        # Initialize ResNet34
        self.resnet34 = models.resnet34()
        # Remove the last two layers
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-2])
        self.ln_reg = ln_reg

        # Multihead Attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               batch_first=True)

        # Adaptive average pooling to maintain compatibility
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(embed_dim, n_classes)

        # Add Linear Regression Layer if used
        if self.ln_reg:
            self.Reg = nn.Linear(n_classes, 1)

    def freeze_backbone(self):
        for param in self.resnet34.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet34(x)  # Extract features
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Reshape to (batch_size, num_patches, channels)
        x, _ = self.attention(x, x, x)  # Apply self-attention
        x = x.permute(0, 2, 1).view(batch_size, channels, height, width)  # Reshape back
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        if self.ln_reg:
            x = self.Reg(x)
        return x


class MULTIPLE_CNN_HEAD(nn.Module):
    def __init__(self, n_classes, out_features=256, ln_reg=None):
        super().__init__()

        # Initialize resnet
        self.resnet34 = models.resnet34()
        in_features = self.resnet34.fc.in_features
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-2])
        self.ln_reg = ln_reg

        # Create multiple convolutional layers with different kernel sizes and specified output features
        kernel_sizes = [(9, 19), (19, 9), (9, 9)]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_features, out_features, kernel_size=k, padding=[ki//2 for ki in k]) for k in kernel_sizes
        ])

        self.relu = nn.ReLU(inplace=True)  # ReLU activation

        # Add Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Add the Flatten layer
        self.flatten = nn.Flatten()

        # Adjust Linear layer to account for different numbers of convolutional outputs
        self.Linear = nn.Linear(out_features * len(kernel_sizes), n_classes)

        # Add Linear Regression Layer if used
        if self.ln_reg:
            self.Reg = nn.Linear(n_classes, 1)

    def freeze_backbone(self):
        for param in self.resnet34.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet34(x)

        # Apply each convolution and concatenate their outputs
        x = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1)

        # Global Average Pooling and Flatten
        x = self.gap(x)
        x = self.flatten(x)

        # Classification head
        x = self.Linear(x)

        if self.ln_reg:
            x = self.Reg(x)
        return x
# ------------------------------------------------------------------------------ #

# --------------------------- Functions and Methods ---------------------------- #


class AbidImageDataset(Dataset):
    def __init__(self, dataset_json_file, transform=None, target_transform=None, shuffle=None):
        # Load Data from json file
        with open(dataset_json_file, 'r') as f:
            self.data = json.load(f)

        # Load data transform
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_abid_img(idx: int):
        """
        Retrieves a PIL Image from an image index.
        """
        s3 = boto3.client('s3')
        bucket_name = "aft-vbi-pds"
        object_key = f"bin-images/{idx}.jpg"
        n_tries = 10
        for tr in range(n_tries):
            try:
                response = s3.get_object(Bucket=bucket_name, Key=object_key)
                image_data = response['Body'].read()
                pil_image = Image.open(BytesIO(image_data))
                return pil_image
            except Exception as e:
                logger.info(f"Error retrieving image from S3: {e}")
                time.sleep(2)
                continue

        return None

    def __getitem__(self, idx):
        # Retrieve images and labels
        if self.shuffle:
            random.shuffle(self.data[idx])

        images = [self.get_abid_img(x[0]) for x in self.data[idx]]
        labels = [x[1] for x in self.data[idx]]

        # Apply transformations to images
        if self.transform:
            images = [self.transform(img) for img in images]
        else:
            images = [torch.tensor(img) for img in images]

        # Stack images into a single tensor
        # Shape: [batch_size, channels, height, width]
        images = torch.stack(images)

        # Apply transformations to labels
        if self.target_transform:
            labels = [self.transform(label) for label in labels]
        else:
            labels = [torch.tensor(label) for label in labels]

        # Stack label into a single tensor
        labels = torch.stack(labels)  # Shape: [batch_size]
        return images, labels


def get_model(model_name, n_classes, ln_reg=None, cuda=None,
              checkpoint_path=None):
    if model_name == "DNN_HEAD":
        model = DNN_HEAD(n_classes, ln_reg=ln_reg)
    elif model_name == "CNN_HEAD":
        model = CNN_HEAD(n_classes, ln_reg=ln_reg)
    elif model_name == "ATT_HEAD":
        model = ATT_HEAD(n_classes, ln_reg=ln_reg)
    elif model_name == "MULTIPLE_CNN_HEAD":
        model = MULTIPLE_CNN_HEAD(n_classes, ln_reg=ln_reg)
    else:
        return None

    logger.info(f"{checkpoint_path}")
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            logger.info(f"Loading Checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            return None

    if ln_reg:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if cuda:
        model.cuda()
        criterion = criterion.cuda()

    model.freeze_backbone()

    return model, criterion


def get_data_loaders(train_data_json: str, valid_data_json: str):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
            ])
    valid_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
            ])

    # Check we have received correct path
    logger.info("Checking data file path")
    is_f_train = os.path.isfile(train_data_json)
    is_f_valid = os.path.isfile(valid_data_json)
    if not is_f_train or not is_f_valid:
        logger.info("Data file path not found")
        logger.info(f"Training Data path: {train_data_json}")
        logger.info(f"Validation Data path: {valid_data_json}")
        return None

    train_dt = AbidImageDataset(train_data_json,
                                transform=train_transform,
                                shuffle=True)
    valid_dt = AbidImageDataset(valid_data_json,
                                transform=valid_transform,
                                shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(train_dt,
                                                    batch_size=None,
                                                    shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid_dt,
                                                    batch_size=None,
                                                    shuffle=True)

    return train_data_loader, valid_data_loader


def transform_input_inference(x):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tx = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
            ])
    return tx(x)


def train(model, criterion, optimizer, lr, data_loader, ln_reg=None, cuda=None):
    model.train()
    mse_loss = nn.MSELoss()

    # Training model
    for i, (x, yg) in enumerate(data_loader):
        # Send data to gpu
        if cuda:
            x = x.cuda()
            yg = yg.cuda()

        # Forward propagation
        yp = model(x)
        if ln_reg:
            yg = yg.float()
            yp = yp.squeeze(1)
        loss = criterion(yp, yg)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate metric on static values
    model.eval()
    metrics = {'acc': 0, 'mse': 0, 'loss': 0}

    for i, (x, yg) in enumerate(data_loader):
        # Send data to gpu
        if cuda:
            x = x.cuda()
            yg = yg.cuda()

        # Forward propagation
        with torch.no_grad():
            yp = model(x)
            if ln_reg:
                yg = yg.float()
                yp = yp.squeeze(1)

            loss = criterion(yp, yg)
            metrics['loss'] = metrics['loss'] + loss.item()

        # Process output
        if ln_reg:
            predicted = torch.round(yp).long()
            yg = yg.long()
        else:
            score, predicted = torch.max(yp, dim=1)

        # Accuracy
        correct = float((yg == predicted).sum().item())
        total_samples = x.size(0)
        acc = (correct / total_samples)
        metrics['acc'] = metrics['acc'] + acc
        # RMSE
        mse = mse_loss(predicted.float(), yg.float()).item()
        metrics['mse'] = metrics['mse'] + mse

    for mtr in metrics.keys():
        metrics[mtr] = metrics[mtr]/len(data_loader)
        logger.info(f"{mtr}: {metrics[mtr]}")

    return metrics


def validate(model, criterion, data_loader, ln_reg=None, cuda=None):
    model.eval()
    mse_loss = nn.MSELoss()

    metrics = {'acc': 0, 'mse': 0, 'loss': 0}

    for i, (x, yg) in enumerate(data_loader):
        # Send data to gpu
        if cuda:
            x = x.cuda()
            yg = yg.cuda()

        # Forward propagation
        with torch.no_grad():
            yp = model(x)
            if ln_reg:
                yg = yg.float()
                yp = yp.squeeze(1)
            loss = criterion(yp, yg)
            metrics['loss'] = metrics['loss'] + loss.item()

        # Process output
        if ln_reg:
            predicted = torch.round(yp).long()
            yg = yg.long()
        else:
            score, predicted = torch.max(yp, dim=1)

        # Accuracy
        correct = float((yg == predicted).sum().item())
        total_samples = x.size(0)
        acc = (correct / total_samples)
        metrics['acc'] = metrics['acc'] + acc
        # RMSE
        mse = mse_loss(predicted.float(), yg.float()).item()
        metrics['mse'] = metrics['mse'] + mse

    for mtr in metrics.keys():
        metrics[mtr] = metrics[mtr]/len(data_loader)
        logger.info(f"{mtr}: {metrics[mtr]}")

    return metrics


def run(model_name, n_classes, ln_reg, checkpoint_path, epochs, lr, evaluate,
        train_data_json, valid_data_json, new_checkpoint_path):
    # Get model
    cuda = torch.cuda.is_available()
    if not cuda:
        logger.info("Not cuda found defaulting to CPU")

    logger.info("Loading Model")
    model, criterion = get_model(model_name,
                                 n_classes,
                                 ln_reg,
                                 cuda,
                                 checkpoint_path)

    logger.info("Creating data loaders")
    train_loader, valid_loader = get_data_loaders(train_data_json,
                                                  valid_data_json)
    metrics = {}
    if evaluate:
        logger.info("Validation Stage")
        metrics['val'] = validate(model, criterion, valid_loader, ln_reg, cuda)
        return metrics

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch+1}")
        logger.info("Training Stage")
        metrics['train'] = train(model,
                                 criterion,
                                 optimizer,
                                 lr,
                                 train_loader,
                                 ln_reg,
                                 cuda)
        logger.info("Validation Stage")
        metrics['val'] = validate(model,
                                  criterion,
                                  valid_loader,
                                  ln_reg,
                                  cuda)
    logger.info("Saving Checkpoint")
    torch.save({
        "arch": model_name,
        "state_dict": model.state_dict(),
    }, new_checkpoint_path)

    return metrics
# ------------------------------------------------------------------------------ #

# Setting cli tool and task


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--type_of_inference', type=str)
    parser.add_argument('--checkpoint_name', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--data', type=str,
                        default=os.environ["SM_CHANNEL_PROJECT_DATA"])
    parser.add_argument('--output_data', type=str,
                        default=os.environ["SM_OUTPUT_DATA_DIR"])

    args = parser.parse_args()
    logger.info("Starting Script")
    logger.info(args)

    if args.type_of_inference == "linear_regression":
        ln_reg = True
    else:
        ln_reg = False

    logger.info("Directories")
    logger.info(f"{os.listdir(args.data)}")
    data_dir = os.path.join(args.data, 'data')
    train_data = os.path.join(data_dir, "counting_cleaned_train.json")
    valid_data = os.path.join(data_dir, "counting_cleaned_valid.json")
    name = f"{args.model_name}_{datetime.now()}_{uuid4()}"
    logger.info(f"Checkpoint name: {name}")
    checkpoint_path = os.path.join(args.data, "snapshots", args.checkpoint_name)
    new_checkpoint_path = os.path.join(args.output_data, f"{name}.pth.tar")
    log_path = os.path.join(args.output_data, f"{name}.json")

    logger.info("Starting Job")
    metrics = run(args.model_name,
                  args.n_classes,
                  ln_reg,
                  checkpoint_path,
                  args.epochs,
                  args.learning_rate,
                  args.evaluate,
                  train_data,
                  valid_data,
                  new_checkpoint_path)

    if metrics:
        metrics['lr'] = args.learning_rate
        metrics['model_name'] = args.model_name
        metrics['n_classes'] = args.n_classes
        metrics['type_of_inference'] = args.type_of_inference
        metrics['epochs'] = args.epochs
        metrics['initial_checkpoint_weights'] = args.checkpoint_name
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)


