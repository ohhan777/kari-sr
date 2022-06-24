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
                fake_lr_filename = os.path.join(fake_lr_save_dir, os.path.basename(filename[j]).replace('.tif','.png'))
                if os.path.exists(fake_lr_filename):
                    fake_lr_filename = fake_lr_filename.replace('.png','_1.png')
                img = fake_y[j].mul(255.0).clamp(0,255).cpu().numpy().squeeze(0).astype(np.uint8)
                cv2.imwrite(fake_lr_filename, img)
                
                hr_filename = os.path.join(hr_save_dir, os.path.basename(filename[j]).replace('.tif','.png'))
                if os.path.exists(hr_filename):
                    hr_filename = hr_filename.replace('.png','_1.png')
                img = hr_x[j].mul(255.0).clamp(0,255).cpu().numpy().squeeze(0).astype(np.uint8)
                cv2.imwrite(hr_filename, img)



if __name__ == "__main__":
    device = select_device('', 16)
    from models.gans import Generator
    G_XY = Generator().to(device)
    # Hyperparameters
    hyp = str(ROOT / 'data/hyps/hyp.wv3-k3a.yaml')
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))   # https://pytorch.org/docs/stable/elastic/run.html
    RANK = int(os.getenv('RANK', -1))
    workers = 16
    import yaml
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    from utils.dataloader import create_dataloader
    train_loader, train_dataset = create_dataloader(is_train=True, batch_size=16,
                                                    hyp=hyp, augment=True, cache=False, rank=LOCAL_RANK, workers=workers)
    weights = './runs/exp4/weights/best.pt'
    save_dir = Path(weights).parent.parent
    run('./runs/exp4/weights/best.pt', device, G_XY, train_loader, save_dir)