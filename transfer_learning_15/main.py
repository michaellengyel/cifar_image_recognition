# New Topics:
# - ImageFolder
# - Scheduler
# - Transfer Learning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    data_dir = 'data/hymenoptera_data'
    sets = ['train', 'val']
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0) for x in sets}
    dataset_sizes = {x: len(image_datasets[x]) for x in sets}
    class_names = image_datasets['train'].classes
    print(class_names)

    def train_model(model, criterion, optimizer, scheduler, num_epochs, visualization):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs-1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in sets:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluation mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over coco
                for inputs, labels in dataloaders[phase]:

                    if visualization:

                        out = torchvision.utils.make_grid(inputs)
                        out = out.permute(1, 2, 0).numpy()

                        print(labels)
                        plt.imshow(out)
                        plt.show()

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward
                    # track history if only training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model

    model = models.resnet18(pretrained=True)

    number_of_params = 0
    # Freezing all pre-trained layers in the ResNet18
    for param in model.parameters():
        number_of_params += 1
        param.requires_grad = False

    num_of_features = model.fc.in_features
    print(num_of_features)

    model.fc = nn.Linear(num_of_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=5, visualization=True)


if __name__ == "__main__":
    main()

