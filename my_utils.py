import torch
import pickle
import os

from models import SequenceModel, get_faster_rcnn

def load_pickle(filename):
    with open(filename, 'rb') as file:
        temp = pickle.load(file )
        train_dataset = temp['train_dataset']
        test_dataset = temp['test_dataset']
    del temp
    
    return train_dataset, test_dataset

def save_model(path, name, arch, model, epochs, optimizer, criterion):
    model_path = os.path.join(path, name) + '.tar'
    
    torch.save({
        'model' : arch,
        'epoch' : epochs,
        'model_state_dict': model.state_dict(),
        'optimizer' : optimizer,
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion' : criterion
        }, model_path)
        
def load_model(filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    if checkpoint['model'] == 'rcnn':
        model = get_faster_rcnn()
    else:
        model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    epoch = checkpoint['epoch']
    model.to(device)

    return model, optimizer, criterion, epoch