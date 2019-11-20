import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from random import shuffle
import time

from model import SVHNModel
from torch_dataset import SVHNDataset
from utils import *

class Trainer:
    def __init__(self, args):
        self.opt = args
        self.data_path = self.opt.data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = self.opt.model_path
        self.model_name = self.opt.model_name

        # Training parameters
        self.input_size = (self.opt.height, self.opt.width)
        self.batch_size = int(self.opt.batch_size)
        self.num_workers = int(self.opt.num_workers)
        self.epochs = int(self.opt.num_epochs)
        self.lr = float(self.opt.learning_rate)
        self.step = int(self.opt.scheduler_step_size)

        # Create model and place on GPU
        self.model = SVHNModel()
        self.model = self.model.to(self.device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.scheduler = StepLR(self.optimizer, step_size=self.step, gamma=0.5)

        print('Training options:\n'
              '\tInput size: {}\n\tBatch size: {}\n\tEpochs: {}\n\t'
              'Learning rate: {}\n\tStep Size: {}\n\tLoss: {}\n\tOptimizer: {}\n'. \
              format(self.input_size, self.batch_size, self.epochs, self.lr, self.step, self.criterion, self.optimizer))

        # load data from pickle file
        train_data, temp_data = load_pickle(os.path.join(self.data_path, 'SVHN_metadata.pickle'))
        shuffle(train_data)
        shuffle(temp_data)

        # Splitting the data to create a validation set
        split = round(0.5 * len(temp_data))
        validation_data = temp_data[split:]
        test_data = temp_data[:split]

        print('Training on:\n'
              '\tTrain files: {}\n\tValidation files: {}\n\tTest files: {}\n' \
              .format(len(train_data), len(validation_data), len(test_data)))

        # Data transformations to be used during loading of images
        self.data_transforms = {
            'Train': transforms.Compose([transforms.RandomRotation(0.2),
                                         transforms.transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                         transforms.Resize(self.input_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                              std=[0.5, 0.5, 0.5])
                                         ]),
            'Validation': transforms.Compose([transforms.RandomRotation(0.2),
                                              transforms.transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                              transforms.Resize(self.input_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ]),
            'Test': transforms.Compose([transforms.Resize(self.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])])}

        # Creating PyTorch datasets
        self.datasets = dict()
        self.datasets['Train'] = SVHNDataset(train_data, os.path.join(self.data_path, 'train'), \
                                             self.data_transforms['Train'])

        self.datasets['Validation'] = SVHNDataset(validation_data, os.path.join(self.data_path, 'test'), \
                                                  self.data_transforms['Validation'])

        self.datasets['Test'] = SVHNDataset(test_data, os.path.join(self.data_path, 'test'), \
                                            self.data_transforms['Test'])

        # Creating PyTorch dataloaders
        self.dataloaders = {i: DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True, \
                                          num_workers=self.num_workers) for i in ['Train', 'Validation']}

        self.test_loader = DataLoader(dataset=self.datasets['Test'], batch_size=1, shuffle=True)

    def calc_loss(self, length, digit1, digit2, digit3, digit4, digit5, gt_length, gt_labels):
        length_loss = self.criterion(length, gt_length - 1)
        digit1_loss = self.criterion(digit1, gt_labels[:, 0])
        digit2_loss = self.criterion(digit2, gt_labels[:, 1])
        digit3_loss = self.criterion(digit3, gt_labels[:, 2])
        digit4_loss = self.criterion(digit4, gt_labels[:, 3])
        digit5_loss = self.criterion(digit5, gt_labels[:, 4])

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

        num_digits_correct += (digit1_preds.eq(gt_labels[:, 0]) +
                               digit2_preds.eq(gt_labels[:, 1]) +
                               digit3_preds.eq(gt_labels[:, 2]) +
                               digit4_preds.eq(gt_labels[:, 3]) +
                               digit5_preds.eq(gt_labels[:, 4])).cpu().sum()

        return num_seq_correct.item(), num_digits_correct.item()

    def run_epoch(self, phase):
        running_loss = 0.0
        running_seq_corrects = 0
        running_digit_corrects = 0
        running_total_digits = 0

        if phase == 'Train':
            self.model.train()
        else:
            self.model.eval()

        # Looping through batches
        for i, (images, gt_lengths, gt_labels) in enumerate(self.dataloaders[phase]):

            # Ensure we're doing this calculation on our GPU if possible
            images = images.to(self.device)
            gt_lengths = gt_lengths.to(self.device)
            gt_labels = gt_labels.to(self.device)

            # Zero parameter gradients
            self.optimizer.zero_grad()

            # Calculate gradients only if we're in the training phase
            with torch.set_grad_enabled(phase == 'Train'):

                # This calls the forward() function on a batch of inputs
                length, digit1, digit2, digit3, digit4, digit5 = self.model(images)

                # Calculate the loss of the batch
                loss = self.calc_loss(length, digit1, digit2, digit3, digit4, digit5, gt_lengths, gt_labels)

                # Adjust weights through backprop if we're in training phase
                if phase == 'Train':
                    loss.backward()
                    self.optimizer.step()

            # Calculate sequence-wise and digit-wise accuracy for the batch
            seq_correct, digit_correct = self.calc_acc(digit1, digit2, digit3, digit4, digit5, gt_labels)

            # Document statistics for the batch
            running_loss += loss.item() * images.size(0)
            running_seq_corrects += seq_correct
            running_digit_corrects += digit_correct

        self.scheduler.step()

        # Calculate epoch statistics
        epoch_loss = running_loss / self.datasets[phase].__len__()
        seq_acc = running_seq_corrects / self.datasets[phase].__len__()
        digit_acc = running_digit_corrects / self.datasets[phase].__len__() * 5

        return epoch_loss, seq_acc, digit_acc

    def train(self):
        start = time.time()

        best_model_wts = self.model.state_dict()
        best_acc = 0.0

        print('| Epoch\t | Train Loss\t| Train Seq Acc\t| Train Dig Acc\t| Valid Loss\t| Valid Seq Acc\t| Valid Dig Acc\t| Epoch Time |')
        print('-' * 118)

        # Iterate through epochs
        for epoch in range(self.epochs):

            epoch_start = time.time()

            # Training phase
            train_loss, train_seq_acc, train_dig_acc = self.run_epoch('Train')

            # Validation phase
            val_loss, val_seq_acc, val_dig_acc = self.run_epoch('Validation')

            epoch_time = time.time() - epoch_start

            # Print statistics after the validation phase
            print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.0f}m {:.0f}s     |"
                  .format(epoch + 1, train_loss, train_seq_acc, train_dig_acc, val_loss, val_seq_acc, val_dig_acc, epoch_time // 60, epoch_time % 60))

            # Copy and save the model's weights if it has the best accuracy thus far
            if val_seq_acc > best_acc:
                best_acc = val_seq_acc
                best_model_wts = self.model.state_dict()

        total_time = time.time() - start

        print('-' * 118)
        print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
        print('Best validation accuracy: {:.4f}'.format(best_acc))

        # load best model weights and save them
        self.model.load_state_dict(best_model_wts)

        save_model(self.model_path, self.model_name, self.model, self.epochs, self.optimizer, self.criterion)

        return