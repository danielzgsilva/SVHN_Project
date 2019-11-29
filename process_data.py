import tarfile
import os
import sys
import h5py

from options import SVHN_Options
from my_utils import *

options = SVHN_Options()
opts = options.parse()

class PreProcessor():
    def __init__(self, options):
        self.opt = options
        self.data_path = self.opt.data_path

    def extract_data(self, file_name, data_root):
        # remove .tar.gz
        folder = os.path.splitext(os.path.splitext(file_name)[0])[0]
        if os.path.isdir(folder):
            print('{} already present - Skipping extraction of {}'.format(folder, file_name))

        else:
            print('Extracting {}'.format(file_name))
            with tarfile.open(file_name) as tar:
                sys.stdout.flush()
                tar.extractall(data_root)
                tar.close()

    def unpack_box(self, idx, struct):
        """ Get labels, left, top, width, height of each bounding box in an image """

        attributes = ['label', 'height', 'width', 'top', 'left']

        # Instantiate metadata dict for each image
        metadata = {attr: [] for attr in attributes}

        # This function retreives each data point corresponding to a bounding box
        def get_attrs(attr, obj):
            vals = []
            if obj.shape[0] == 1:
                vals.append(int(obj[0][0]))
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(struct[obj[k][0]][0][0]))

            metadata[attr] = vals

        # Get bounding box metadata for an image
        box = struct['/digitStruct/bbox'][idx]
        struct[box[0]].visititems(get_attrs)

        return metadata

    def unpack_filename(self, idx, struct):
        ''' Get the filename of an image'''

        filename = struct['/digitStruct/name']
        return ''.join([chr(v[0]) for v in struct[filename[idx][0]]])

    def unpack_struct(self, filename):
        # Read in H5PY file
        struct = h5py.File(filename)
        num_files = struct['/digitStruct/name'].size

        print('Unpacking {} files from {}'.format(num_files, filename))

        # Digit 10 used to represent no digit
        digits = [10, 10, 10, 10, 10]

        # List to hold the metadata dict of each image in a dataset
        dataset = []

        # Loop through each image file
        for i in range(num_files):

            # Get bounding box metadata for each image
            metadata = self.unpack_box(i, struct)

            metadata['length'] = len(metadata['label'])

            # Skip examples with more than 5 digits in it
            if metadata['length'] > 5:
                continue

            # Digit 10 will be used to represent no digit
            digits = [10, 10, 10, 10, 10]
            for idx, label in enumerate(metadata['label']):
                digits[idx] = int(label if label != 10 else 0)

            metadata['digits'] = digits

            # Get filename for each image
            metadata['filename'] = self.unpack_filename(i, struct)

            if i % 2500 == 0 or i == num_files - 1:
                print('Image {}/{}'.format(i, num_files))

            # Add this images metadata to the dataset
            dataset.append(metadata)

        return dataset

    def save_pickle(self, filename, data, test_data):
        with open(filename, 'wb') as file:
            metadata = {'train_dataset': data, 'test_dataset': test_data}
            pickle.dump(metadata, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def process(self):
        self.extract_data(os.path.join(self.data_path, 'train.tar.gz'), self.data_path)
        self.extract_data(os.path.join(self.data_path, 'test.tar.gz'), self.data_path)

        num_train_files = len(os.listdir(os.path.join(self.data_path, 'train')))
        num_test_files = len(os.listdir(os.path.join(self.data_path, 'test')))
        print('Training: {} Testing: {}'.format(num_train_files, num_test_files))

        #num_extra_files = len(os.listdir(os.path.join(data_root, 'extra')))
        #print('Training: {} Testing: {} Extra: {}'\
         #     .format(num_train_files, num_test_files, num_extra_files))

        data = self.unpack_struct(os.path.join(self.data_path, 'train', 'digitStruct.mat'))
        test_data = self.unpack_struct(os.path.join(self.data_path, 'test', 'digitStruct.mat'))

        metadata_file = 'SVHN_metadata.pickle'
        print('Creating {} file'.format(metadata_file))
        self.save_pickle(os.path.join(self.data_path, metadata_file), data, test_data)

        print('Done preprocessing!')

if __name__ == "__main__":
    processor = PreProcessor(opts)
    processor.process()