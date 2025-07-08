import os
import packaging.version as pkg
import torch
from threading import Thread
from utils.plots import plot_images, plot_samples
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.general import colorstr

LOGGERS = ('csv', 'tb', 'wandb')
RANK = int(os.getenv('RANK', -1))

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse(wandb.__version__) >= pkg.parse('0.12.2') and RANK in [0, -1]:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None


class Loggers():
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = ['D loss', # D loss
                     'G loss', # G loss
                     ]  # params
        self.best_keys = ['best/epoch', 'best/total_loss']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize kari-seg 🚀 runs (RECOMMENDED)"
        
        # W&B
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

    def on_train_start(self):
        # Callback runs on train start
        pass


    def on_train_epoch_start(self, epoch):
        # Callback runs on train epoch start
        pass

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_train_batch_end(self, epoch, ni, real_x, fake_y, cycle_x, real_y, fake_x, cycle_y):
        # Callback runs on train batch end                
        if ni == 0:
            filename = str(self.save_dir / f'train_{epoch}.png')
            Thread(target=plot_samples, args=(real_x, fake_y, cycle_x, real_y, fake_x, cycle_y, filename), daemon=True).start() 
    
    def on_fit_epoch_end(self, vals, epoch, lowest_loss, cl):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            if lowest_loss == cl:
                self.wandb.wandb_run.summary['best/epoch'] = epoch   # log best results in the summary
                self.wandb.wandb_run.summary['best/total_loss'] = cl   # log best results in the summary
            self.wandb.log(x)
            # upload images
            files = sorted(self.save_dir.glob('*.png'))
            self.wandb.log({"Fake LR": [wandb.Image(str(f), caption=f.name) for f in files]})
            self.wandb.end_epoch(best_result=lowest_loss == cl)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, epoch, results):
        if self.wandb:
            self.wandb.finish_run()

    def on_val_end(self):
        # Callback runs on val end
        pass
        





