from options import SVHN_Options
from trainer import Trainer

options = SVHN_Options()
opts = options.parse()
                                      
                                      
if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()