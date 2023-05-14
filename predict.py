import argparse
import torch
import subprocess
import torch.nn.functional as F
from torchvision import models
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T
import json
from datetime import datetime
import os
import glob
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir='flowers'
test_dir= data_dir+'/test'
checkpoint_dir = ''
checkpoint_file = 'model.pth'
#top_k

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, nargs='?', default=checkpoint_file)
    parser.add_argument('-img','--img_path',type=str, default= test_dir + '/69/image_05959.jpg')
    parser.add_argument('-cat','--category_names',dest='category_names', default='cat_to_name.json', type=str)
    parser.add_argument('-k','--top_k',default=1, type=int)
    parser.add_argument('--gpu',dest='gpu', default=False, action='store_true')
    return parser.parse_args()

def loadCheckpoint(path): # Have to change

    if device =='cuda:0':
        cp = torch.load(path)
    else:
        cp = torch.load(path, map_location = lambda storage, loc: storage)
    if cp['arch']== 'resnet50':
        model= models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(
            nn.Linear(2048,cp['hidden_units']),
            nn.ReLU(),
            nn.Dropout(p=0.45),
            nn.Linear(cp['hidden_units'],102)

        )
        model.fc = classifier
        model.load_state_dict(cp['state_Dict'])
    elif(cp['arch']=='resnet34'):
        model = models.resnet34(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(
            nn.Linear(512, cp['hidden_units']),
            nn.ReLU(),
            nn.Dropout(p=0.45),
            nn.Linear(cp['hidden_units'], 102)

        )
        model.fc = classifier
        model.load_state_dict(cp['state_Dict'])
    else:
        class Net(nn.Module):
            def __init__(self, num_classes, hidden_unit, name):
                super().__init__()
                if (name == 'vgg16'):
                    self.featuresVgg = nn.Sequential(
                        *list(models.vgg16(pretrained=True).features.children())
                    )

                else:
                    self.featuresVgg = nn.Sequential(
                        *list(models.vgg13(pretrained=True).features.children())
                    )
                out_features = 1000
                self.classifier = nn.Sequential(
                    nn.Linear(25088, 1000),
                    nn.Linear(1000, hidden_unit),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_unit, num_classes),

                )

            def forward(self, x):

                x = self.featuresVgg(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

        model = Net(num_classes=102, hidden_unit=cp['hidden_units'], name=cp['arch'])
        model.load_state_dict(cp['state_Dict'])
    model.class_to_idx = cp['class_to_idx']
    return model

def process_image(image):
    imag=Image.open(image)
    ratio = max(224/imag.size[0], 224/imag.size[1])
    resize_values = (int(imag.size[0]*ratio)), int(imag.size[1]*ratio)
    imag=imag.resize(resize_values, Image.ANTIALIAS)

    width, height = imag.size
    l=(width-224)/2
    r= (width+224)/2
    t=(height-224)/2
    b=(height+224)/2
    imag=imag.crop((l,t,r,b))
    image_np = np.array(imag)
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    image_np = (image_np/255 - mean)/std
    image_np = image_np.transpose((2,0,1))
    return torch.from_numpy(image_np)

def predict(image_path, model, top_k=5):
    image=process_image(image_path)
    image=image.unsqueeze_(0)
    image=image.to(device)
    model.type(torch.DoubleTensor)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output=model.forward(image)
        prob=torch.exp(output)
        probs, indices = torch.topk(prob, top_k)
        indice = indices.cpu()
        indice= indice.numpy()[0]
        idx_to_class = {val: key for key,val in model.class_to_idx.items()}
        classes = [idx_to_class[index] for index in indice]
    return probs, classes

    
# if len(glob.glob(checkpoint_dir + '/*pth')) >0:
#     checkpoint = max(glob.glob(checkpoint_dir+"/*.pth"), key=os.path.getctime)
# else:
#     print("No checkpoint saved available")
#     sys.exit(1)

def main():
    args = get_input_args()
    if len(glob.glob(args.checkpoint))==0:
        print("No found check point")
        sys.exit(1)
    if len(glob.glob(args.img_path))==0:
        print("Not found category names")
        sys.exit(1)
    if args.top_k <1:
        print("Invalid top_k")
        sys.exit(1)
    device = torch.device('cuda:0' if args.gpu else 'cpu')
    model = loadCheckpoint(args.checkpoint)
    probs,classes= predict(args.img_path,model,top_k = args.top_k)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        name = [cat_to_name[i] for i in classes]
        print('Class names: ', name)
    print('Classes: ', classes)
    print('Prob: ', probs)


if __name__ == '__main__':
    install("Image")
    main()
