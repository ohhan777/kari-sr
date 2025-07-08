import os
import numpy as np
import cv2
import torch
from pathlib import Path
from utils.callbacks import Callbacks
from utils.general import (print_args, LOGGER, colorstr, one_cycle, increment_path,
                           check_yaml, methods, check_suffix, init_seeds, intersect_dicts,
                           strip_optimizer, get_latest_run, FullModel, AverageMeter, check_version)
from utils.torch_utils import is_main_process, select_device, time_sync, reduce_tensor
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  

@torch.no_grad()
def run(weights=None,
        device='',
        G_XY=None,
        dataloader=None,
        save_dir=Path(''),
        ):

        # load the best weight
        ckpt = torch.load(weights, map_location='cpu')

        csd = ckpt['G_XY'].float().state_dict()
        csd = intersect_dicts(csd, G_XY.state_dict())
        G_XY.load_state_dict(csd, strict=False)

        # make dir
        fake_lr_save_dir = save_dir/ 'pairs' / 'fake_lr'
        fake_lr_save_dir.mkdir(parents=True, exist_ok=True)
        hr_save_dir = save_dir / 'pairs' / 'hr'
        hr_save_dir.mkdir(parents=True, exist_ok=True)

        G_XY.eval()
        for i, (ds_x, _, hr_x, filename) in enumerate(dataloader):      # x: downsampled hr image, y: lr image
            ds_x = ds_x.to(device)     # Nx1x512x512
            fake_y = G_XY(ds_x)

            # save the generated image and its hr image pair
            for j in range(fake_y.size(0)):
                fake_lr_filename = os.path.join(fake_lr_save_dir, os.path.basename(filename[j]).replace('.tif', '_000.tif'))
                if os.path.exists(fake_lr_filename):
                    number = int(fake_lr_filename[-7:-4]) + 1
                    str_number = '%03d' % number 
                    fake_lr_filename = fake_lr_filename[:-7] + str_number + '.tif'
                img = fake_y[j].mul(16383).clamp(0,16383).cpu().numpy().squeeze(0).astype(np.uint16)
                #img = fake_y[j].mul(255).clamp(0,255).cpu().numpy().squeeze(0).astype(np.uint8)
                cv2.imwrite(fake_lr_filename, img)
                
                hr_filename = os.path.join(hr_save_dir, os.path.basename(filename[j]).replace('.tif','_000.tif'))
                if os.path.exists(hr_filename):
                    number = int(hr_filename[-7:-4]) + 1
                    str_number = '%03d' % number 
                    hr_filename = hr_filename[:-7] + str_number + '.tif'
                img = hr_x[j].mul(16383).clamp(0,16383).cpu().numpy().squeeze(0).astype(np.uint16)
                #img = hr_x[j].mul(255).clamp(0,255).cpu().numpy().squeeze(0).astype(np.uint8)
                cv2.imwrite(hr_filename, img)



if __name__ == "__main__":
    batch_size = 32
    device = select_device('', batch_size)
    from models.gans import Generator
    G_XY = Generator().to(device)
    # Hyperparameters
    #hyp = str(ROOT / 'data/hyps/hyp.k3a-k3.yaml')
    hyp = str(ROOT / 'data/hyps/hyp.wv3-k3.yaml')
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))   # https://pytorch.org/docs/stable/elastic/run.html
    RANK = int(os.getenv('RANK', -1))
    workers = batch_size
    import yaml
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    from utils.dataloader import create_dataloader
    train_loader, train_dataset = create_dataloader(is_train=True, batch_size=batch_size,
                                                    hyp=hyp, augment=True, cache=False, rank=LOCAL_RANK, workers=workers)
    #weights = './runs/exp58/weights/best.pt' #WV3-K3A
    weights = './runs/exp67/weights/best.pt' #K3A-K3
    save_dir = Path(weights).parent.parent
    run(weights, device, G_XY, train_loader, save_dir)