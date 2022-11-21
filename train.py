import json
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import argparse
import time

def arguments():
    parse = argparse.ArgumentParser()

    parse.add_argument('data_dir') 
    parse.add_argument('--save_dir',type=str,default='checkpoint.pth')
    parse.add_argument('--arch',type=str,default='densenet121')
    parse.add_argument('--learning_rate',type=float,default=0.003)
    parse.add_argument('--hidden_units',type=int,default=512)
    parse.add_argument('--epochs',type=int,default=1)
    parse.add_argument('--gpu',action='store_true')

    return parse.parse_args()

def print_command_lines(in_arg):
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\n     data_dir =",in_arg.data_dir,"\n     save_dir =", in_arg.save_dir, 
              "\n     arch =", in_arg.arch, "\n     learning_rate =", in_arg.learning_rate,
              "\n     hidden_Units =", in_arg.hidden_units, "\n     epochs =",in_arg.epochs,
              "\n     Device =",'cuda' if in_arg.gpu else 'cpu')

def training(arch,data_directory,save_dir,epochs,learn_rate,hidden_units,device):

    densenet121=models.densenet121(pretrained=True)
    vgg19=models.vgg19(pretrained=True)

    models_ = {'densenet121':densenet121, 'vgg19':vgg19}
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                        ])

    cost_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                        ])
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    cost_data = datasets.ImageFolder(valid_dir, transform=cost_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(cost_data, batch_size=32)

    image_datasets = [train_data, cost_data]

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    class_to_idx = dict()
    for i in cat_to_name:
        class_to_idx[cat_to_name[i]]=int(i)

    model = models_[arch]

    for each in model.parameters():
        each.require_grad=False
    
    if(arch == 'densenet121'):
        classifier = nn.Sequential(nn.Linear(1024,hidden_units),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(hidden_units,int(hidden_units/2)),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(int(hidden_units/2),102),
                            nn.LogSoftmax(dim=1))
    else:
        classifier = nn.Sequential(nn.Linear(25088,1024),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(1024,hidden_units),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(hidden_units,int(hidden_units/2)),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(int(hidden_units/2),102),
                            nn.LogSoftmax(dim=1))
    

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = learn_rate)
    
    model.to(device)
    train_loss = 0

    print("....................................Training.................................... \n")
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    else:
        model.cpu()
        
    train_loss = 0
    start = time.time()

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            if cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in valid_loader:

                    if cuda:
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}..\n "
                  f"Train loss: {train_loss/len(train_loader):.3f}.. "
                  f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
            train_loss = 0
            model.train()


    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

    model.class_to_idx = image_datasets[0].class_to_idx

    checkpoint = {'arch': arch,
                'learning_rate': 0.003,
                'batch_size': 64,
                'classifier' : classifier,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_dir)


in_args = arguments()
print_command_lines(in_args)
device_ = 'cuda' if in_args.gpu else 'cpu'
training(in_args.arch,in_args.data_dir,in_args.save_dir,in_args.epochs,in_args.learning_rate,in_args.hidden_units,device_)

