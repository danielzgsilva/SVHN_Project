import os
import argparse

file_dir = os.path.dirname(__file__)

class SVHN_Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SVHN options")

        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, 'SVHN_dataset'))
        self.parser.add_argument("--model_path",
                                 type=str,
                                 help="directory to save model weights in",
                                 default=os.path.join(file_dir, "models"))

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="name of the model")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=64)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=128)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=32)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=0.001)
        self.parser.add_argument("--num_epochs",
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        self.parser.add_argument("--model_to_load",
                                 nargs="+",
                                 type=str,
                                 help="model to load")


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
