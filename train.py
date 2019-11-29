from options import SVHN_Options
from trainer import Trainer, DetectionTrainer

options = SVHN_Options()
opts = options.parse()

if __name__ == "__main__":
    trainer = DetectionTrainer(opts)
    trainer.train()