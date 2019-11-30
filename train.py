from options import SVHN_Options
from trainers import ClassificationTrainer, DetectionTrainer

options = SVHN_Options()
opts = options.parse()

if __name__ == "__main__":
    if opts.model_type == 'detector':
        trainer = DetectionTrainer(opts)
    else:
        trainer = ClassificationTrainer(opts)

    trainer.train()