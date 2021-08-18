import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from sklearn.model_selection import KFold


def train_model(model, device, dataloaders, criterion, optimizer, num_samples, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / num_samples[phase]
            epoch_acc = running_corrects.double(
            ) / num_samples[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet":
        """
        Resnet50
        """
        model_ft = models.resnet50(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """
        Alexnet
        """
        model_ft = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """
        VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        """
        Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """
        Densenet
        """
        model_ft = models.densenet121(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def select_device(device='0'):
    # device = 'cpu' or '0'
    cpu = device.lower() == 'cpu'
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        print("Number of cuda devices:", n)
        assert n > int(device) and int(
            device) >= 0, f'Invalid device {device} requested'

    return torch.device(f'cuda:{int(device)}' if cuda else 'cpu')


def main(opt):
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = opt.model

    k_folds = opt.k_folds
    data_dir = opt.dataset
    num_classes = opt.classes
    batch_size = opt.batch
    num_epochs = opt.epochs
    feature_extract = opt.freeze

    input_size = 299 if (model_name == "inception") else 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size, input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create datasets
    dataset = datasets.ImageFolder(data_dir, data_transforms['train'])

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Detect if we have a GPU available
    device = select_device(opt.device)

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs
    ohist, kfold_res = [], []

    plt.title(opt.model)
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'\n\nFOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        dataloaders_dict = {}
        dataloaders_dict["train"] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler)
        dataloaders_dict["valid"] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=val_subsampler)
        
        num_samples = {
            "train": len(train_subsampler),
            "valid": len(val_subsampler)
        }

        # Initialize the model for this run
        model = initialize_model(model_name, num_classes, feature_extract)
        # Send the model to GPU
        model = model.to(device)

        # Gather the parameters to be optimized/updated in this run
        params_to_update = model.parameters()
        if feature_extract:
            params_to_update = []
            for _, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model, hist = train_model(model, device, dataloaders_dict, criterion,
            optimizer, num_samples, num_epochs=num_epochs, is_inception=(model_name=="inception"))

        # Save trained weights
        file_name = f"{opt.model}__{time.strftime('%m-%d-%Y_%H-%M-%S')}"
        if not os.path.isdir('./weights'):
            os.mkdir('./weights')
        torch.save(model.state_dict(), os.path.join(
            './weights', f"{file_name}_fold{fold}.pt"))

        ohist = [h.cpu().numpy() for h in hist]
        plt.plot(range(1, num_epochs+1), ohist, label=f"Fold {fold}")
        plt.ylim((0, 1.))
        plt.xticks(np.arange(1, num_epochs+1, 25))

        kfold_res.append(max(hist).cpu().numpy())

    # Save the plot
    if not os.path.isdir('./histograms'):
        os.mkdir('./histograms')
    plt.legend()
    plt.savefig(os.path.join('./histograms', file_name + ".png"))
    print("kfold_res", kfold_res)
    print("\nAver. acc:", np.mean(kfold_res))


if __name__ == "__main__":
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    parser = argparse.ArgumentParser(
        description='Train a classifier with Pytorch')
    parser.add_argument('--model', type=str, default='resnet',
                        help='Name of the model (resnet, alexnet, vgg, squeezenet, densenet)')
    parser.add_argument('--dataset', type=str,
                        default='data/wbc', help='Top level data directory')
    parser.add_argument('--classes', type=int,
                        default=5, help='Number of classes in the dataset')
    parser.add_argument('--batch', type=int,
                        default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int,
                        default=25, help='Number of epochs to train for')
    parser.add_argument('--freeze', action='store_true',
                        help='Set True to update the reshaped layer params only')
    parser.add_argument('--k-folds', type=int,
                        default=5, help='Number of folds to validation')
    parser.add_argument('--device', default='0',
                        help='Cuda device, i.e. 0 or 1 or ...')
    opt = parser.parse_args()
    print(opt)
    main(opt)
