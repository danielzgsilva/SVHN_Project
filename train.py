import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

from random import shuffle
import time

from model import SVHN_CNN
from torch_dataset import SVHN_Dataset
from options import SVHN_Options
from utils import *

options = SVHN_Options()
opts = options.parse()

class Trainer():
    def __init__(self, options):
        self.opt = options
        self.data_path = self.opt.data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = self.opt.model_path
        self.model_name = self.opt.model_name
       
        self.input_size = (self.opt.height, self.opt.width)
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers
        self.epochs = self.opt.num_epochs
        self.lr = self.opt.learning_rate
        
        # load data from pickle file
        data, test_data = load_pickle(os.join.path(self.data_path, 'SVHN_metadata.pickle'))
        shuffle(data)
        shuffle(test_data)
                                      
        # Splitting the data to create a validation set
        split = round(0.85 * len(data))
        train_data = data[:split]
        validation_data = data[split:]
        
        print('Train files: {} Validation files: {} Test files: {}'\
              .format(len(train_data), len(validation_data), len(test_data)))
                                                                      
        self.data_transforms = {
            'Train': transforms.Compose([transforms.Resize(self.input_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
            ]),
            'Validation': transforms.Compose([transforms.Resize(self.input_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
            ]),
            'Test': transforms.Compose([transforms.Resize(self.input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])])}
                                      
                                      
        self.datasets = dict()
        self.datasets['Train'] = SVHN_Dataset(train_data, os.path.join(self.data_path, 'train'),\
                                              self.data_transforms['Train'])
                                      
        self.datasets['Validation'] = SVHN_Dataset(validation_data, os.path.join(self.data_path, 'train'),\
                                                   self.data_transforms['Validation'])
                                      
        self.datasets['Test'] = SVHN_Dataset(test_data, os.path.join(self.data_path, 'test'),\
                                             self.data_transforms['Test'])
        
        self.dataloaders = {i: DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True,\
                                          num_workers = self.num_workers) for i in ['Train', 'Validation']}

        self.test_loader = DataLoader(dataset = self.datasets['Test'], batch_size = 1, shuffle=True)

    def calc_loss(self, criterion, length, digit1, digit2, digit3, digit4, digit5, gt_length, gt_labels):
        length_loss = criterion(length, gt_length)
        digit1_loss = criterion(digit1, gt_labels[:, 0])
        digit2_loss = criterion(digit2, gt_labels[:, 1])
        digit3_loss = criterion(digit3, gt_labels[:, 2])
        digit4_loss = criterion(digit4, gt_labels[:, 3])
        digit5_loss = criterion(digit5, gt_labels[:, 4])

        loss = length_loss + digit1_loss + digit2_loss + digit3_loss + digit4_loss + digit5_loss
        return loss
                                      
    def calc_acc(self, digit1, digit2, digit3, digit4, digit5, gt_labels):
        # Get predictions
        _, digit1_preds = torch.max(digit1, 1)
        _, digit2_preds = torch.max(digit2, 1)
        _, digit3_preds = torch.max(digit3, 1)
        _, digit4_preds = torch.max(digit4, 1)
        _, digit5_preds = torch.max(digit5, 1)

        num_seq_correct = 0
        num_digits_correct = 0

        num_seq_correct += (digit1_preds.eq(gt_labels[:, 0]) &
                            digit2_preds.eq(gt_labels[:, 1]) &
                            digit3_preds.eq(gt_labels[:, 2]) &
                            digit4_preds.eq(gt_labels[:, 3]) &
                            digit5_preds.eq(gt_labels[:, 4])).cpu().sum()

        num_digits_correct = digit1_preds.eq(gt_labels[:, 0]).cpu().sum() + \
                            digit2_preds.eq(gt_labels[:, 1]).cpu().sum() + \
                            digit3_preds.eq(gt_labels[:, 2]).cpu().sum() + \
                            digit4_preds.eq(gt_labels[:, 3]).cpu().sum() + \
                            digit5_preds.eq(gt_labels[:, 4]).cpu().sum() 

        return num_seq_correct.item(), num_digits_correct.item()
              
    def run_epoch(self, model, criterion, optimizer, dataloaders, device, phase):
        running_loss = 0.0
        running_seq_corrects = 0
        running_digit_corrects = 0

        if phase == 'Train':
            model.train()
        else:
            model.eval()

        # Looping through batches
        for i, (images, gt_lengths, gt_labels) in enumerate(dataloaders[phase]):

            # Ensure we're doing this calculation on our GPU if possible
            images = images.to(device)
            gt_lengths = gt_lengths.to(device)
            gt_labels = gt_labels.to(device)

            # Zero parameter gradients
            optimizer.zero_grad()

            # Calculate gradients only if we're in the training phase
            with torch.set_grad_enabled(phase == 'Train'):

                # This calls the forward() function on a batch of inputs
                length, digit1, digit2, digit3, digit4, digit5 = model(images)

                # Calculate the loss of the batch
                loss = self.calc_loss(criterion, length,\
                                 digit1, digit2, digit3, digit4, digit5, gt_lengths, gt_labels)

                # Adjust weights through backpropagation if we're in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

            # Calculate sequence-wise and digit-wise accuracy for the batch
            seq_correct, digit_correct = self.calc_acc(digit1, digit2, digit3,\
                                                  digit4, digit5, gt_labels)

            # Document statistics for the batch
            running_loss += loss.item() * images.size(0)
            running_seq_corrects += seq_correct
            running_digit_corrects += digit_correct

        # Calculate epoch statistics
        epoch_loss = running_loss / self.datasets[phase].__len__()
        epoch_acc = running_seq_corrects / self.datasets[phase].__len__()

        return epoch_loss, epoch_acc
    
    def train(self, model, criterion, optimizer, num_epochs, dataloaders, device):
        start = time.time()

        best_model_wts = model.state_dict()
        best_acc = 0.0

        print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t| Epoch Time |')
        print('-' * 86)

        # Iterate through epochs
        for epoch in range(num_epochs):

            epoch_start = time.time()

            # Training phase
            train_loss, train_acc = self.run_epoch(model, criterion, optimizer, dataloaders, device, 'Train')

            # Validation phase
            val_loss, val_acc = self.run_epoch(model, criterion, optimizer, dataloaders, device, 'Validation')

            epoch_time = time.time() - epoch_start

            # Print statistics after the validation phase
            print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.0f}m {:.0f}s     |"
                          .format(epoch + 1, train_loss, train_acc, val_loss, val_acc, epoch_time // 60, epoch_time % 60))

            # Copy and save the model's weights if it has the best accuracy thus far
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = model.state_dict()

        total_time = time.time() - start

        print('-' * 74)
        print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
        print('Best validation accuracy: {:.4f}'.format(best_acc))

        # load best model weights and return them
        model.load_state_dict(best_model_wts)

        return model
    
    def start_train(self):
        model = SVHN_CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = self.lr)
        model.to(self.device)
                                      
        model = self.train(model, criterion, optimizer, self.epochs, self.dataloaders, self.device)
                                      
        save_model(self.model_path, self.model_name, model, self.epochs, optimizer, criterion)
                                      
                                      
if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.start_train()