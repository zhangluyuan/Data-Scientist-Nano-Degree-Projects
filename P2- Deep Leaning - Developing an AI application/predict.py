import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from train import build_model

import json
import argparse
from torchvision import models

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='File path of the image to predict')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file containing label names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = parser.parse_known_args()

# this method process image before predictions
def process_image(file_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # load in the image
    image = Image.open(file_path)
    image.load()

    # resize the image so that the short side is 256 pixels
    old_size = np.array(image.size)
    new_size = old_size/(min(old_size))*256
    new_size = tuple((new_size).astype(int))
    image = image.resize(new_size)

    # crop the image to size 224x224
    width, height = image.size

    left = int(width/2) - 112
    right = left + 224
    lower = int(height/2) - 112
    upper = lower + 224

    image = image.crop((left, lower, right, upper))

    # convert pixel values to 0-1
    np_image = np.asarray(image, dtype = 'int32') / 255

    # perform standard scaling on the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image

# this method loads the pre-trained model
def load_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    model = getattr(models, arch)(pretrained=True)

    #freeze the parameters of the model
    for param in model.parameters():
        param.requires_grad=False

    if 'resnet' in arch or 'inception' in arch:
        model.fc = checkpoint['classifier']
    else:
        model.classifier = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


# Implement the code to predict the class from an image file
def predict(image, checkpoint, topk=5, labels='', gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use command line values when specified
    if args.image:
        image = args.image

    if args.checkpoint:
        checkpoint = args.checkpoint

    if args.topk:
        topk = args.topk

    if args.labels:
        labels = args.labels

    if args.gpu:
        gpu = args.gpu

    # Load the trained from from checkpoint
    model = load_model(checkpoint)

    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        model.cuda()

    was_training = model.training
    model.eval()

    # process image
    image = process_image(image)

    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0) # this is for VGG

    if gpu and torch.cuda.is_available():
        image = image.cuda()

    result = model(image).topk(topk)

    if torch.cuda.is_available():
        probs = (result[0].data).cpu().exp().numpy()[0]
        indx = result[1].data.cpu().numpy()[0]
    else:
        probs = (result[0].data).exp().numpy()[0]
        indx = result[1].data.numpy()[0]
        
    class_to_indx = model.class_to_idx
    # reverse map indecies to classes
    indx_to_class = {class_to_indx[c]:c for c in class_to_indx}
    probs = probs.tolist()
    indx = indx.tolist()
    classes = [indx_to_class[i] for i in indx]

    if labels:
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]

    model.train(mode=was_training)

    # Print only when invoked by command line
    if args.image:
        print('Predictions and probabilities:', list(zip(classes, probs)))

    return probs, classes

# Perform predictions if invoked from command line
if args.image and args.checkpoint:
    predict(args.image, args.checkpoint)
