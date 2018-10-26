from PIL import Image
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import torch
import numpy as np
import os

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

# define commanline Argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Data directory')
parser.add_argument('--gpu', action='store_true', help='Use GPU is available')
parser.add_argument('--checkpoint', type=str, help='Directory to save checkpoint')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of nodes of hidden layer')
parser.add_argument('--epochs', type=int, help='Number of epochs')

args, _ = parser.parse_known_args()

# laod and build the model
def build_model(arch, hidden_units, num_labels):
    '''build a new model based to a pre-trained network'''
    # load a pre-trained network
    try:
        model = getattr(models, arch)(pretrained=True)
    except:
        raise ValueError('Invalid architecture', arch)

    if 'resnet' in arch or 'inception' in arch:
        input_units = model.fc.in_features
    else:
        try:
            input_units = model.classifier[0].in_features
        except:
            input_units = model.classifier.in_features

    if input_units <= hidden_units and input_units > num_labels*2:
        hidden_units = num_labels*2

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad=False

    # Define a new classifier
    classifier=nn.Sequential(OrderedDict([('do1', nn.Dropout()),
                                        ('fc1', nn.Linear(input_units, hidden_units)),
                                        ('relu1', nn.ReLU()),
                                        ('do2', nn.Dropout()),
                                        ('fc2', nn.Linear(hidden_units, num_labels)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))

    # Replace model's old classifier or fc with the new classifier
    if 'resnet' in arch or 'inception' in arch:
        model.fc=classifier
    else:
        model.classifier=classifier
    return model

# this method transform the data_sets
def data_transformer(data_dir):
    '''This method loads and transforms image data, and return the transformed datasets'''
    '''input: a string, the directory that contains data'''
    set_folders = os.listdir(data_dir)
    data_dirs = {e: data_dir + '/' + e for e in set_folders}

    # Define transforms for the training, validation, and testing sets
    data_transforms = {}
    for e in set_folders:
        if e =='train':
            data_transforms[e] = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        else:
            data_transforms[e] = transforms.Compose([transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])


    # Load and transform the datasets with ImageFolder
    data_sets = {e: datasets.ImageFolder(data_dirs[e], transform = data_transforms[e])
                for e in set_folders}

    return data_sets

# this method calculates loss and accuracy on validation and test data
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    total = 0
    for images, labels in testloader:
        if gpu and torch.cuda.is_available():
            images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        total +=1

    return test_loss/total, accuracy/total

# train the model
def train_model(image_datasets, arch='vgg19', hidden_units=4096, epochs=4, learning_rate=0.001, gpu=True, checkpoint='checkpoint'):
    # Use command line values when specified
    if args.arch:
        arch = args.arch
    if args.hidden_units:
        hidden_units = args.hidden_units
    if args.epochs:
        epochs = args.epochs
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.gpu:
        gpu = args.gpu
    if args.checkpoint:
        checkpoint = args.checkpoint

    # define data loaders
    data_loaders = {e: DataLoader(image_datasets[e], batch_size=64, shuffle=True)
                    for e in image_datasets.keys()}

    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)

    # build and load the model
    num_labels = len(image_datasets['train'].classes)
    model = build_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")

    # define criterion and optimizer
    criterion=nn.NLLLoss()
    if 'resnet' in arch or 'inception' in arch:
        optimizer=optim.Adam(model.fc.parameters(), lr=0.001)
    else:
        optimizer=optim.Adam(model.classifier.parameters(), lr=0.001)

    # train the model
    print_every=40
    steps=0

    for e in range(epochs):
        running_loss=0
        for ii, (inputs, labels) in enumerate(data_loaders['train']):
            steps += 1
            if gpu and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # Forward and backward passes
            outputs=model.forward(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                if 'valid' in data_loaders:
                    # Make sure network is in eval mode for inference
                    model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        valid_loss, accuracy = validation(model, data_loaders['valid'], criterion)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(valid_loss),
                          "Validation Accuracy: {:.3f}".format(accuracy))
                    model.train()
                else:
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every))
                running_loss = 0

    # test the models
    model.eval()
    dataiter = iter(data_loaders['test'])
    images, labels = dataiter.next()
    if gpu and torch.cuda.is_available():
        images, labels = images.to('cuda'), labels.to('cuda')

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(images)

    ps = torch.exp(output)
    equality = (labels.data == ps.max(dim=1)[1])
    accuracy = equality.type(torch.FloatTensor).mean()

    print('test_accuracy: {:.3f}'.format(accuracy))

    # save checkpoint if requested
    if checkpoint:
        model.class_to_idx = image_datasets['train'].class_to_idx
        try:
            classifier = model.classifier
        except:
            classifier = model.fc
        checkpoint_dict = {'class_to_idx': model.class_to_idx,
                            'arch': arch,
                            'epochs':epochs,
                            'hidden_units':hidden_units,
                            'learning_rate': 0.001,
                            'state_dict': model.state_dict(),
                            'classifier': classifier,
                            'optimizer_state': optimizer.state_dict
                        }
        torch.save(checkpoint_dict, checkpoint)
        print('model checkpoint is saved to ', checkpoint)

    # return the model
    return model

# Train model if invoked from command line
if args.data_dir:
    image_datasets = data_transformer(args.data_dir)
    train_model(image_datasets)
