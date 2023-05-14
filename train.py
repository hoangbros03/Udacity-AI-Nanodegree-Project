# Importing necessary libs
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
import random
import os
import argparse
import sys
import inspect
import torch.optim as optim

# Important variables with default values. Change if needed
data_dir = 'flowers'
save_dir = 'model.pth'
arch = 'resnet50'
learning_rate = 0.01
hidden_units = 512
epochs = 20
gpu = False
cat_to_name = {'21': 'fire lily', '3': 'canterbury bells', '45': 'bolero deep blue', '1': 'pink primrose',
               '34': 'mexican aster', '27': 'prince of wales feathers', '7': 'moon orchid', '16': 'globe-flower',
               '25': 'grape hyacinth', '26': 'corn poppy', '79': 'toad lily', '39': 'siam tulip', '24': 'red ginger',
               '67': 'spring crocus', '35': 'alpine sea holly', '32': 'garden phlox', '10': 'globe thistle',
               '6': 'tiger lily', '93': 'ball moss', '33': 'love in the mist', '9': 'monkshood',
               '102': 'blackberry lily', '14': 'spear thistle', '19': 'balloon flower', '100': 'blanket flower',
               '13': 'king protea', '49': 'oxeye daisy', '15': 'yellow iris', '61': 'cautleya spicata',
               '31': 'carnation', '64': 'silverbush', '68': 'bearded iris', '63': 'black-eyed susan',
               '69': 'windflower', '62': 'japanese anemone', '20': 'giant white arum lily', '38': 'great masterwort',
               '4': 'sweet pea', '86': 'tree mallow', '101': 'trumpet creeper', '42': 'daffodil',
               '22': 'pincushion flower', '2': 'hard-leaved pocket orchid', '54': 'sunflower', '66': 'osteospermum',
               '70': 'tree poppy', '85': 'desert-rose', '99': 'bromelia', '87': 'magnolia', '5': 'english marigold',
               '92': 'bee balm', '28': 'stemless gentian', '97': 'mallow', '57': 'gaura', '40': 'lenten rose',
               '47': 'marigold', '59': 'orange dahlia', '48': 'buttercup', '55': 'pelargonium',
               '36': 'ruby-lipped cattleya', '91': 'hippeastrum', '29': 'artichoke', '71': 'gazania',
               '90': 'canna lily', '18': 'peruvian lily', '98': 'mexican petunia', '8': 'bird of paradise',
               '30': 'sweet william', '17': 'purple coneflower', '52': 'wild pansy', '84': 'columbine',
               '12': "colt's foot", '11': 'snapdragon', '96': 'camellia', '23': 'fritillary', '50': 'common dandelion',
               '44': 'poinsettia', '53': 'primula', '72': 'azalea', '65': 'californian poppy', '80': 'anthurium',
               '76': 'morning glory', '37': 'cape flower', '56': 'bishop of llandaff', '60': 'pink-yellow dahlia',
               '82': 'clematis', '58': 'geranium', '75': 'thorn apple', '41': 'barbeton daisy', '95': 'bougainvillea',
               '43': 'sword lily', '83': 'hibiscus', '78': 'lotus lotus', '88': 'cyclamen', '94': 'foxglove',
               '81': 'frangipani', '74': 'rose', '89': 'watercress', '73': 'water lily', '46': 'wallflower',
               '77': 'passion flower', '51': 'petunia'}


def get_dir(data_dir):
    train_dir = data_dir + "/train"
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return (data_dir, train_dir, valid_dir, test_dir)


def get_transforms(image_size=[224, 244], random_rotation=40, mean_normalize=[0.485, 0.456, 0.406],
                   std_normalize=[0.229, 0.224, 0.225]):
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(224),
        T.RandomRotation(random_rotation),
        T.ToTensor(),
        T.Normalize((mean_normalize[0], mean_normalize[1], mean_normalize[2]),
                    (std_normalize[0], std_normalize[1], std_normalize[2]))
    ])


def get_testTransforms(image_size=[224, 244], mean_normalize=[0.485, 0.456, 0.406], std_normalize=[0.229, 0.224, 0.225],
                       center_crop=224):
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(center_crop),
        T.ToTensor(),
        T.Normalize((mean_normalize[0], mean_normalize[1], mean_normalize[2]),
                    (std_normalize[0], std_normalize[1], std_normalize[2]))
    ])


def get_datasets(paths, transform_train, transform_test):
    data, train, valid, test = paths
    image_datasets = torchvision.datasets.ImageFolder(root=train, transform=transform_train)
    image_validsets = torchvision.datasets.ImageFolder(root=valid, transform=transform_test)
    image_testsets = torchvision.datasets.ImageFolder(root=test, transform=transform_test)
    return (image_datasets, image_validsets, image_testsets)


def get_dataLoaders(batch_size, shuffle, paths, transform_train, transform_test):
    train, valid, test = get_datasets(paths, transform_train, transform_test)

    dataLoaders_train = DataLoader(dataset=train, batch_size=batch_size, shuffle=shuffle)
    dataLoaders_valid = DataLoader(dataset=valid, batch_size=batch_size, shuffle=shuffle)
    dataLoaders_test = DataLoader(dataset=test, batch_size=batch_size, shuffle=shuffle)
    return (dataLoaders_train, dataLoaders_valid, dataLoaders_test)


def get_pretrainedModel(arch):
    models_list = ['resnet50', 'resnet34', 'vgg16', 'vgg13']

    # if arch not in model_dict:
    #     #If not found, use resnet50
    #     model = models.resnet50(pretrained=True)
    #     return
    # else:
    #     result = model_dict[arch]
    #     model = result(pretrained=True)
    out_features = 0
    if (arch == models_list[0]):
        model = models.resnet50(pretrained=True)
        out_features = 2048
    elif (arch == models_list[1]):
        model = models.resnet34(pretrained=True)
        out_features = 512
    elif (arch == models_list[2]):
        model = models.vgg16(pretrained=True)
        out_features = model.classifier[-1].out_features
    else:
        model = models.vgg13(pretrained=True)
        out_features = model.classifier[-1].out_features

    for p in model.parameters():
        p.requires_grad = False
    return model, out_features


# # Get number of output classes
# num_classes = model.fc.out_features

def get_forwardNN(out_features, num_classes=102, hidden_unit=500):
    class Net(nn.Module):
        def __init__(self, num_classes, hidden_unit, out_features):
            super().__init__()
            self.hl1 = nn.Linear(out_features, hidden_unit)
            self.hl2 = nn.Linear(hidden_unit, num_classes)
            self.dropout = nn.Dropout(p=0.5)
            self.ReLU = nn.ReLU()

        def forward(self, x):
            x = self.ReLU(self.hl1(x))
            x = self.dropout(x)
            x = self.hl2(x)
            return x

    net = Net(num_classes, hidden_unit, out_features)
    return net


def get_addForwardModelResnet(pretrainedModel, afterModel, modelName):
    if (modelName == 'resnet50' or modelName == 'resnet34'):
        pretrainedModel.fc = afterModel
        return pretrainedModel


def get_device(gpu=torch.cuda.is_available()):
    if (gpu):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def Valid(model, criterion, valid_dataloader, device=get_device()):
    loss = 0.0
    accuracy = 0.0
    len_dataset = 0.0
    model.to(device)
    for images, labels in valid_dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equity = (labels == ps.max(dim=1)[1])
        len_dataset += labels.size(0)
        accuracy += equity.type(torch.FloatTensor).mean()
    return loss, accuracy, len_dataset


def Train(model, optimizer, criterion, train_dataloaders, valid_dataloaders, epochs=epochs, device=get_device(),
          lr=learning_rate):
    running_loss = 0
    step = 0
    model.to(device)
    for i in range(epochs):
        model.train()
        for images, labels in train_dataloaders:
            step += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 30 == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy, len_dataset = Valid(model, criterion, valid_dataloaders, device)

                print(
                    "Step: {}, epoch: {}, loss: {}, test_loss: {}, accuracy: {}".format(step, i + 1, running_loss / 30,
                                                                                        test_loss / len(
                                                                                            valid_dataloaders), (
                                                                                                accuracy / len(
                                                                                            valid_dataloaders))))
                running_loss = 0
                model.train()

    print("Done")


def Accuracy(model, criterion, dataloaders_test, device=get_device()):
    model.eval()
    with torch.no_grad():
        loss, accuracy, len_dataset = Valid(model, criterion, dataloaders_test, device)
    print("Accuracy on test set: {}".format(accuracy / len(dataloaders_test)))
    return accuracy / len(dataloaders_test)


def saveCheckpoint(name, model, arch, hidden_units, image_datasets):
    # Remember to take the out_features first!
    model.class_to_idx = image_datasets.class_to_idx
    modelcp = {
        'state_Dict': model.state_dict(),
        'arch': arch,
        'hidden_units': hidden_units,
        'class_to_idx': model.class_to_idx
    }
    torch.save(modelcp, name)
    print("Save successfully!")


# def loadCheckpoint(path, device= get_device()):
#     if device=='cuda:0':
#         cp=torch.load(path)
#     else:
#         cp=torch.load(path, map_location = lambda storage, loc: storage)
#     model = cp['model']
#     for param in model.parameters():
#         param.requires_grad=False
#     model.fc=cp['classifier']
#     model.load_state_dict(cp['state_Dict'])
#     epochs = cp['epochs']
#     model.class_to_idx = cp['class_to_idx']
#     return model

# GET NEED PARSER HERE

def get_VggModel(out_features, num_classes=102, hidden_unit=hidden_units, name='vgg16'):
    class Net(nn.Module):
        def __init__(self, num_classes, hidden_unit, out_features, name):
            super().__init__()
            if (name == 'vgg16'):
                self.featuresVgg = nn.Sequential(

                    *list(models.vgg16(pretrained=True).features.children())
                )
            else:

                self.featuresVgg = nn.Sequential(

                    *list(models.vgg13(pretrained=True).features.children())
                )
            self.classifier = nn.Sequential(
                nn.Linear(25088, out_features),
                nn.Linear(out_features, hidden_unit),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_unit, num_classes),

            )

        def forward(self, x):

            x = self.featuresVgg(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    net = Net(num_classes, hidden_unit, out_features, name)
    return net


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, nargs='?', default='flowers')
    parser.add_argument('--save_dir', type=str, default=save_dir)
    parser.add_argument('--arch', dest='arch', default=arch, choices=['resnet50', 'resnet34', 'vgg16', 'vgg13'])
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', default=learning_rate, type=float)
    parser.add_argument('-dout', '--dropout', dest='dropout', default=0.5, type=float)
    parser.add_argument('-hu', '--hidden_units', dest='hidden_units', default=hidden_units, type=int)
    parser.add_argument('-e', '--epochs', dest='epochs', default=epochs, type=int)
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true')
    return parser.parse_args()


def main(data_dir=data_dir, save_dir=save_dir, arch=arch, learning_rate=learning_rate, hidden_units=hidden_units,
         epochs=epochs, gpu=gpu, cat_to_name=cat_to_name):
    args = get_input_args()
    epochs = args.epochs
    data_dir = args.data_dir
    arch = args.arch
    learning_rate = args.learning_rate
    gpu = args.gpu
    hidden_units = args.hidden_units
    data_dir, train_dir, valid_dir, test_dir = get_dir(data_dir=data_dir)

    dataLoaders_train, dataLoaders_valid, dataLoaders_test = get_dataLoaders(batch_size=32, shuffle=True, paths=(
        data_dir, train_dir, valid_dir, test_dir), transform_train=get_transforms(),
                                                                             transform_test=get_testTransforms())
    ptModel, out_features = get_pretrainedModel(arch=arch)
    print(out_features)
    if (arch == 'resnet50' or arch == 'resnet34'):

        model = get_addForwardModelResnet(ptModel, afterModel=get_forwardNN(out_features, num_classes=102,
                                                                            hidden_unit=hidden_units), modelName=arch)
        optimizer = optim.Adam(model.fc.parameters())
    else:
        model = get_VggModel(out_features, num_classes=102, hidden_unit=hidden_units, name=arch)
        optimizer = optim.Adam(model.classifier.parameters())

    Train(model, optimizer=optimizer, lr=learning_rate, criterion=nn.CrossEntropyLoss(),
          train_dataloaders=dataLoaders_train, valid_dataloaders=dataLoaders_valid, epochs=epochs,
          device=get_device(gpu=gpu))
    acu = Accuracy(model, nn.CrossEntropyLoss(), dataloaders_test=dataLoaders_test, device=get_device(gpu=gpu))
    image_datasets, v_v, t_v = get_datasets(get_dir(data_dir=data_dir),get_transforms(),get_testTransforms())
    saveCheckpoint(save_dir, model, arch=arch, hidden_units=hidden_units, image_datasets=image_datasets)


if __name__ == "__main__":
    main(data_dir=data_dir, save_dir=save_dir, arch=arch, learning_rate=learning_rate, hidden_units=hidden_units,
         epochs=epochs, gpu=gpu, cat_to_name=cat_to_name)





