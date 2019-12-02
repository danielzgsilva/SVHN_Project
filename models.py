import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class SequenceModel(nn.Module):
    def __init__(self):
        super(SequenceModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.4)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.5)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.2)
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.2)
        )

        self.block8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.2)
        )

        # self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(192 * 2 * 4, 3072),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.number_length = nn.Sequential(nn.Linear(3072, 5))
        self.digit1 = nn.Sequential(nn.Linear(3072, 11))
        self.digit2 = nn.Sequential(nn.Linear(3072, 11))
        self.digit3 = nn.Sequential(nn.Linear(3072, 11))
        self.digit4 = nn.Sequential(nn.Linear(3072, 11))
        self.digit5 = nn.Sequential(nn.Linear(3072, 11))

    # Feed the input through each of the layers we defined
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Need to flatten activations before feeding into fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        length = self.number_length(x)
        digit1 = self.digit1(x)
        digit2 = self.digit2(x)
        digit3 = self.digit3(x)
        digit4 = self.digit4(x)
        digit5 = self.digit5(x)

        return length, digit1, digit2, digit3, digit4, digit5

class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            #nn.Dropout(0.2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            #nn.Dropout(0.2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            #nn.Dropout(0.2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            #nn.Dropout(0.2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            #nn.Dropout(0.2)
        )


    # Feed the input through each of the layers we defined
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x

class DetectionModel(nn.Module):
    ''' Creates a Faster RCNN model using the feature extraction model above, DigitModel, as a backbone.
    FasterRCNN implementation from here: https://github.com/pytorch/vision/tree/master/torchvision/models/detection'''
    def __init__(self):
        super(DetectionModel, self).__init__()
        # Creating the feature extractor
        self.feature_extractor = DigitModel()
        self.feature_extractor.out_channels = 512

        # The region proposal network will generate 5x3 anchors per spatial location
        # Each with 5 different sizes and 3 different aspect ratios
        self.anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # Extract and align features from each proposed region on each scale.
        # These features would then be sent to the final fully connected layers for classification and bbox regression
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                        output_size=7,
                                                        sampling_ratio=2)

        # We define 11 classes, 1 for each digit 0-9 and an additional class for the background
        self.num_classes = 11

        # Compile these modules within a PyTorch implementation of Faster RCNN
        self.detector = FasterRCNN(self.feature_extractor,
                                    num_classes=self.num_classes,
                                    rpn_anchor_generator=self.anchor_generator,
                                    box_roi_pool=self.roi_pooler,
                                    min_size=384, max_size=512,
                                    box_score_thresh=0.5, box_nms_thresh=0.5, box_detections_per_img=5)


def get_faster_rcnn():
    model = DetectionModel()
    return model.detector


