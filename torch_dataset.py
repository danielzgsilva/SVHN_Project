import torch
from torch.utils.data import Dataset
import PIL.Image as pil
import os

class SVHNDataset(Dataset):
    """Abstract PyTorch dataset class for the Street View House Numbers dataset"""

    def __init__(self, metadata, data_path, transform=None):
        
        # Metadata is an array of dicts with keys:
        # filename, label, height, width, top, left
        self.metadata = metadata
        self.data_path = data_path
        
        # Transformations to apply to each image 
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        # Load the image
        img = pil.open(os.path.join(os.path.join(self.data_path, self.metadata[idx]['filename'])))

        # Index the metadata array to grab this images bounding box info
        meta = self.metadata[idx]
        
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(meta['length']), torch.tensor(meta['digits'])


class DetectionDataset(Dataset):
    '''
    Abstract PyTorch dataset class for the Street View House Numbers dataset
    Returns a dictionary containing the following:
        boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format
        labels (Int64Tensor[N]): the label for each bounding box
        image_id (Int64Tensor[1]): an image identifier, unique between all the images in the dataset
        area (Tensor[N]): The area of the bounding box.
        iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
        '''

    def __init__(self, metadata, data_path, transform=None):

        # Metadata is an array of dicts with keys:
        # filename, label, height, width, top, left
        self.metadata = metadata
        self.data_path = data_path

        # Transformations to apply to each image
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        # Load the image
        file_path = os.path.join(os.path.join(self.data_path, self.metadata[idx]['filename']))
        img = pil.open(file_path)

        # Index the metadata array to grab this images bounding box info
        meta = self.metadata[idx]

        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        # Getting bounding box coordinates for each digit
        boxes = []
        for i in range(meta['length']):
            x_min = meta['left'][i]
            y_min = meta['top'][i]
            x_max = meta['left'][i] + meta['width'][i]
            y_max = meta['top'][i] + meta['height'][i]

            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {}
        target['boxes'] = boxes

        # Labels for each of the bounding boxes
        target['labels'] = torch.tensor(meta['label'], dtype=torch.int64)

        # Image ID
        target['image_id'] = torch.tensor([idx])

        # Areas of each of the bounding boxes
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # No instances of crowded bounding boxes
        target['iscrowd'] = torch.zeros((meta['length'],), dtype=torch.int64)

        return img, target