from curses import has_ic
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def plot_images(imgs, targets, preds, filename):
    # Plot image grid with labels
    if isinstance(imgs, torch.Tensor):
        img = imgs[0].cpu().float().numpy()   # img: (C, H, W)
    if isinstance(targets, torch.Tensor):
        label = targets[0].cpu().numpy()       # label: (H, W) 
        pred = preds[0].cpu().numpy()

    # de-normalize
    img = img.transpose(1,2,0)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img *= std
    img += mean
    img *= 255.0 

    img = img.astype(np.uint8)  # (C, H, W) -> (H, W, C)
    h, w, _ = img.shape
    target_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3)
    pred_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3)

    # target  
    for i in range(19):
        pos = label == i
        target_label[pos] = list(colors(i, bgr=True))
        pos = ((pred == i) & (label != 255))
        pred_label[pos] = list(colors(i, bgr=True))

    target_img = cv2.addWeighted(target_label, 0.3, img, 1, 0)
    target_img = np.concatenate((target_img, np.zeros((h, 2, 3))), axis=1)  # vertical seperate line
    pred_img = cv2.addWeighted(pred_label, 0.3, img, 1, 0)

    img = np.concatenate((target_img, pred_img), axis=1)

    cv2.imwrite(str(filename), img)

def plot_samples(real_x, fake_y, cycle_x, real_y, fake_x, cycle_y, filename):
    # Plot samples
    with torch.no_grad():
        y = [real_x, fake_y, cycle_x, real_y, fake_x, cycle_y]
        h, w = real_x.shape[2:4]
        z = []
        # de-normalize
        for x in y:
            x = x[0]*255
            x = torch.clamp(x, 0, 255).cpu().numpy().squeeze(0).astype(np.uint8)
            z.append(x)

    img = np.zeros((h*2+2, w*3+4), dtype=np.uint8)
    for i, p in enumerate(z):
        h_idx = i // 3
        w_idx = i  % 3
        img[h_idx*(h+2):h_idx*(h+2)+h, w_idx*(w+2):w_idx*(w+2)+w] = p

    cv2.imwrite(str(filename), img)



def save_histogram(img, filename, num_bins=256, range=(0, 1)):    # for debugging
    histogram, bin_edges = np.histogram(img, num_bins, range)
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([range[0], range[1]])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.savefig(filename)

if __name__ == '__main__':
    real_x = torch.rand((4,1,512,512))
    real_y = torch.rand((4,1,512,512))
    fake_x = torch.rand((4,1,512,512))
    fake_y = torch.rand((4,1,512,512))
    cycle_x = torch.rand((4,1,512,512))
    cycle_y = torch.rand((4,1,512,512))
    
    plot_samples(real_x, fake_y, cycle_x, real_y, fake_x, cycle_y, 'test.png')