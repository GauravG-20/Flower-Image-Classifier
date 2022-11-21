import numpy as np
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import time
import argparse
from PIL import Image
import numpy as np

def arguments():
    parse = argparse.ArgumentParser()

    parse.add_argument('image_path',type=str)
    parse.add_argument('load_checkpoint',type=str)
    parse.add_argument('--category_names',type=str,default='cat_to_name.json')
    parse.add_argument('--top_k',type=int,default=5)
    parse.add_argument('--gpu',action='store_true')

    return parse.parse_args()

def print_command_lines(in_arg):
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\n     image_path =",in_arg.image_path,"\n     checkpoint_path =", in_arg.load_checkpoint, 
              "\n     category_names =", in_arg.category_names,
              "\n     Device =",'cuda' if in_arg.gpu else 'cpu')

def load_checkpoint(filename):
    device = 'cuda:0' if in_args.gpu else 'cpu'
    checkpoint = torch.load(filename, map_location=device)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)

def prediction(image_path,checkpoint_dir,catergory_names,topk,device):
    model = load_checkpoint(checkpoint_dir)

    with open(catergory_names, 'r') as f:
        cat_to_name = json.load(f)

    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    else:
        model.cpu()
    
    model.eval()
    image = process_image(image_path)
    
    image = torch.from_numpy(np.array([image])).float()
    
    image = Variable(image)
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data

    prob = torch.topk(probabilities, topk)[0].tolist()[0] 
    index = torch.topk(probabilities, topk)[1].tolist()[0] 
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    
    label = []
    for i in range(topk):
        label.append(cat_to_name[ind[index[i]]])

    print("{:<20} {:<20}".format('Class','Probability'))
    for i in range(topk):
        print("{:<20} {:<20}".format(label[i],prob[i]))

in_args = arguments()
print_command_lines(in_args)
device_ = 'cuda' if in_args.gpu else 'cpu'
prediction(in_args.image_path,in_args.load_checkpoint,in_args.category_names,in_args.top_k,device_)