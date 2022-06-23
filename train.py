import os
import argparse
import time
from datetime import datetime
from copy import deepcopy
import yaml
import numpy as np
from pathlib import Path
import itertools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam

from models.gans import Generator, Discriminator
from utils.dataloader import create_dataloader
from utils.general import (print_args, LOGGER, colorstr, increment_path, check_yaml, methods, check_suffix, init_seeds,
                           check_yaml, intersect_dicts, strip_optimizer, get_latest_run, FullModel, AverageMeter, check_version)
from utils.torch_utils import (select_device, de_parallel, is_main_process)
from utils.callbacks import Callbacks

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))   # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, resume, weights, noval, nosave, workers = Path(opt.save_dir), opt.epochs, opt.batch_size,\
                                                                opt.resume, opt.weights, opt.noval, opt.nosave, opt.workers

    # Directories
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
  
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    if is_main_process():
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir/ 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # TODO: Loggers (W&B)


    # Config
    cuda = device.type != 'cpu'    
    init_seeds(1 + RANK)

    # Models
    check_suffix(weights, '.pt')  # check weights
    G_XY = Generator().to(device)
    G_YX = Generator().to(device)
    D_X = Discriminator().to(device)
    D_Y = Discriminator().to(device)

    # Dataloaders and Datasets
    train_loader, train_dataset = create_dataloader(is_train=True, batch_size=batch_size // WORLD_SIZE,
                                                    hyp=hyp, augment=True, cache=False, rank=LOCAL_RANK, workers=workers)
    val_loader, _ = create_dataloader(is_train=False, batch_size=batch_size // WORLD_SIZE,
                                                    hyp=hyp, augment=True, cache=False, rank=LOCAL_RANK, workers=workers)


    # Adam optimizer
    G_optimizer = Adam(itertools.chain(G_XY.parameters(), G_YX.parameters()), lr=2e-4, betas=(0.5, 0.999))
    D_optimizer = Adam(itertools.chain(D_X.parameters(), D_Y.parameters()), lr=2e-4, betas=(0.5, 0.999))

    start_epoch, lowest_loss = 0, float("inf")
    if resume:
        ckpt = torch.load(weights, map_location='cpu')

        csd = ckpt['G_XY'].float().state_dict()
        csd = intersect_dicts(csd, G_XY.state_dict())
        G_XY.load_state_dict(csd, strict=False)

        csd = ckpt['G_YX'].float().state_dict()
        csd = intersect_dicts(csd, G_YX.state_dict())
        G_YX.load_state_dict(csd, strict=False)

        csd = ckpt['D_X'].float().state_dict()
        csd = intersect_dicts(csd, D_X.state_dict())
        D_X.load_state_dict(csd, strict=False)
        
        csd = ckpt['D_Y'].float().state_dict()
        csd = intersect_dicts(csd, D_Y.state_dict())
        D_Y.load_state_dict(csd, strict=False)

        if is_main_process():
            LOGGER.info(colorstr('yellow', ('Resuming training from %s saved at %s (last epoch %d)') 
                                            % (weights, ckpt['date'], ckpt['epoch'])))

        if ckpt['D_optimizer'] is not None:
            D_optimizer.load_state_dict(ckpt['D_optimizer'])
            lowest_loss = ckpt['lowest_loss']
        
        if ckpt['G_optimizer'] is not None:
            G_optimizer.load_state_dict(ckpt['G_optimizer'])
        
        start_epoch = ckpt['epoch'] + 1
        assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'        
        del ckpt, csd
            



    # loss functions
    gan_loss_fn = nn.MSELoss()
    cycle_loss_fn = nn.L1Loss()
    identity_loss_fn = nn.L1Loss() 

    # DP (Data-Parallel) mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        if is_main_process():
            LOGGER.info("DP mode is enabled, but DDP is preferred for best performance.")
        G_XY = torch.nn.DataParallel(G_XY)
        G_YX = torch.nn.DataParallel(G_YX)
        D_X = torch.nn.DataParallel(D_X)
        D_Y = torch.nn.DataParallel(D_Y)
    
    # DDP mode
    if cuda and RANK != -1:
        if check_version(torch.__version__, '1.11.0'):
            G_XY = DDP(G_XY, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
            G_YX = DDP(G_YX, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
            D_X = DDP(D_X, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
            D_Y = DDP(D_Y, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
        else:
            G_XY = DDP(G_XY, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
            G_YX = DDP(G_YX, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
            D_X = DDP(D_X, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
            D_Y = DDP(D_Y, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)



    # Start training 
    t0 = time.time()
    D_scaler = amp.GradScaler(enabled=cuda)
    G_scaler = amp.GradScaler(enabled=cuda)
    lambda_X = 10
    lambda_Y = 10
    lambda_I = 0.5

    if is_main_process():
        LOGGER.info(f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n' \
                    f"Logging results to {colorstr('bold', save_dir)}\n" \
                    f"Starting training for {epochs} epochs...")
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        t1 = time.time()
        G_XY.train()
        G_YX.train()
        D_X.train()
        D_Y.train()
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        avg_G_loss = AverageMeter()
        avg_D_loss = AverageMeter()
        for i, (x, y) in enumerate(train_loader):      # x: downsampled hr image, y: lr image
            x = x.to(device)     # Nx1x512x512
            y = y.to(device)     # Nx1x512x512
            
            # Update Discriminator (D) -----------------------------------------------
            
            # enable autograd for Discriminator
            for param in D_X.parameters():
                param.requires_grad = True
            
            for param in D_Y.parameters():
                param.requires_grad = True

            with amp.autocast(enabled=cuda):
                fake_y = G_XY(x)        # fake lr image (Nx1x512x512)
                real_y_score = D_Y(y)   # real lr score (Nx1x62x62)
                fake_y_score = D_Y(fake_y.detach())    # fake lr score (Nx1x62x62)
                real_y_loss = gan_loss_fn(real_y_score, torch.ones_like(real_y_score))
                fake_y_loss = gan_loss_fn(fake_y_score, torch.zeros_like(fake_y_score))
                D_y_loss = real_y_loss + fake_y_loss

                fake_x = G_YX(y)        # fake downsampled hr image (Nx1x512x512)
                real_x_score = D_X(x)   # real downsampled hr score (Nx1x62x62)
                fake_x_score = D_X(fake_x.detach())    # fake downsampled hr score (Nx1x62x62)
                real_x_loss = gan_loss_fn(real_x_score, torch.ones_like(real_x_score))
                fake_x_loss = gan_loss_fn(fake_x_score, torch.zeros_like(fake_x_score))
                D_x_loss = real_x_loss + fake_x_loss
                
                D_loss = (D_x_loss + D_y_loss) / 2

            D_optimizer.zero_grad()
            D_scaler.scale(D_loss).backward()
            D_scaler.step(D_optimizer)
            D_scaler.update()

            # Update Generator (G)  ------------------------------------------------

            # disable autograd for Discriminator
            for param in D_X.parameters():
                param.requires_grad = False
            
            for param in D_Y.parameters():
                param.requires_grad = False
            with amp.autocast(enabled=cuda):
                # adversarial loss
                fake_y_score = D_Y(fake_y)
                fake_x_score = D_X(fake_x)
                fake_y_loss = gan_loss_fn(fake_y_score, torch.ones_like(fake_y_score))
                fake_x_loss = gan_loss_fn(fake_x_score, torch.ones_like(fake_x_score))


                # identity loss
                if lambda_I > 0.0:
                    identity_x = G_YX(x)
                    identity_y = G_XY(y)
                    identity_x_loss = identity_loss_fn(identity_x, x) * lambda_X * lambda_I
                    identity_y_loss = identity_loss_fn(identity_y, y) * lambda_Y * lambda_I
                else:
                    identity_x_loss = 0.0
                    identity_y_loss = 0.0        
    

                # cycle loss
                cycle_y = G_XY(fake_x)
                cycle_x = G_YX(fake_y)
                cycle_y_loss = cycle_loss_fn(y, cycle_y) * lambda_Y 
                cycle_x_loss = cycle_loss_fn(x, cycle_x) * lambda_X

                G_loss = fake_x_loss + fake_y_loss + identity_x_loss + identity_y_loss + cycle_x_loss + cycle_y_loss

            G_optimizer.zero_grad()
            G_scaler.scale(G_loss).backward()
            G_scaler.step(G_optimizer)
            G_scaler.update()

            if i == 0:
                from torchvision.utils import save_image
                save_image(fake_y, f'fake_lr_imgs_epoch_{epoch}.png')
               
            avg_G_loss.update(G_loss.item())
            avg_D_loss.update(D_loss.item())

            if i % 5 == 0:
                print('[%d/%d] D_loss: %.4f, G_loss: %.4f' % (i, len(train_loader), avg_D_loss.average(), avg_G_loss.average()))
        if is_main_process():
            cl = avg_G_loss.average() + avg_D_loss.average()
            if cl < lowest_loss:
                lowest_loss = cl
                best_epoch = epoch
                LOGGER.info(colorstr('yellow','bold','[Best so far]'))

            LOGGER.info('Lowest Loss=%.4f (epoch=%d)' % (lowest_loss, best_epoch))

            # Save model
            ckpt = {'epoch': epoch,
                    'lowest_loss': lowest_loss,
                    'G_XY': deepcopy(de_parallel(G_XY)).half(),
                    'G_YX': deepcopy(de_parallel(G_YX)).half(),
                    'D_X': deepcopy(de_parallel(D_X)).half(),
                    'D_Y': deepcopy(de_parallel(D_Y)).half(),
                    'D_optimizer': D_optimizer.state_dict(),
                    'G_optimizer': G_optimizer.state_dict(),
                    'date': datetime.now().isoformat(),
                    }
            
            # Save last, best, periodical checkpoints
            torch.save(ckpt, last)
            if cl == lowest_loss:
                torch.save(ckpt, best)
            if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')   
            del ckpt         

            LOGGER.info(f'Epoch {epoch} completed in {(time.time() - t1):.3f} seconds.')

    if is_main_process():
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')







def main(opt, callbacks=Callbacks()):
    if is_main_process():
        print_args(FILE.stem, opt)
    # Resume
    if opt.resume: # resume an interrupted run
        epochs = opt.epochs
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.weights, opt.resume, opt.epochs = ckpt, True, epochs  # reinstate
    else:
        opt.hyp, opt.weights, opt.project = \
            check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if RANK != -1:
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(RANK)
        device = torch.device('cuda', RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    
    # Train
    train(opt.hyp, opt, device, callbacks)
    
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights.pt', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT/'runs', help='save to project/runs/name')
    parser.add_argument('--name', default='exp', help='save to project/runs/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.wv3-k3a.yaml', help='hyperparameters path')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=16, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--debug', action='store_true', help='debug mode (training is early stopped every epoch)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)