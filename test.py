import numpy as np
from osgeo import gdal
import cv2
import glob
from tqdm import tqdm

def open_geotiff(path):
    ds = gdal.Open(path)
    assert ds != None, f'{path} is not found'
    bands = []
    for i in range(ds.RasterCount):
        bands.append(ds.GetRasterBand(1).ReadAsArray().astype(np.uint16))
    img = np.dstack(bands) if ds.RasterCount > 1 else bands[0]
    return img


path_list_k3a = glob.glob('/mnt/ohhan/storage_ssd/ai/Datasets/SR_exp/K3A/Shanghai/PAN_filtered/*.tif')
path_list_wv3 = glob.glob('/mnt/ohhan/storage_ssd/ai/Datasets/SR_exp/WV3/Shanghai/PAN_filtered/*.tif')

means = []
vars = []
sum = 0
sq_sum = 0
z_sum = 0
z_sq_sum = 0
for path in tqdm(path_list_wv3):
    img = open_geotiff(path) / (2**11 - 1)
    assert img.any() <= 1.0, 'some value exceeds 1.0'
    sum += img[:1024,:1024].sum()
    sq_sum += (img[:1024,:1024] ** 2).sum()

    z = (img - 0.2257752954889022) / np.sqrt(0.004070299363161584)
    z_sum += z[:1024,:1024].sum()
    z_sq_sum += (z[:1024,:1024] ** 2).sum()
    means.append(z.mean())
    vars.append(z.var())

mean = sum/(1024*1024*len(path_list_wv3))
var = sq_sum/(1024*1024*len(path_list_wv3)) - mean**2
z_mean = z_sum/(1024*1024*len(path_list_wv3))
z_var = z_sq_sum/(1024*1024*len(path_list_wv3)) - z_mean**2
print(f'WV3: mean={mean}, var={var}')
print(f'WV3: mean={z_mean}, var={z_var}')
print(f'WV3-mean: mean={np.mean(means)}, var={np.mean(vars)}')

means = []
vars = []
means = []
vars = []
sum = 0
sq_sum = 0
z_sum = 0
z_sq_sum = 0
for path in tqdm(path_list_k3a):
    img = open_geotiff(path) / (2**14 -1)
    sum += img[:1024,:1024].sum()
    sq_sum += (img[:1024,:1024] ** 2).sum()

    z = (img - 0.14595113602669604) / np.sqrt(0.0016996371834978458)
    z_sum += z[:1024,:1024].sum()
    z_sq_sum += (z[:1024,:1024] ** 2).sum()
    means.append(z.mean())
    vars.append(z.var())

mean = sum/(1024*1024*len(path_list_k3a))
var = sq_sum/(1024*1024*len(path_list_k3a)) - mean**2
z_mean = z_sum/(1024*1024*len(path_list_k3a))
z_var = z_sq_sum/(1024*1024*len(path_list_k3a)) - z_mean**2
print(f'K3A: mean={mean}, var={var}')
print(f'K3A: mean={z_mean}, var={z_var}')
print(f'K3A-mean: mean={np.mean(means)}, var={np.mean(vars)}')
#cv2.imwrite('test.png', (img*255.0).astype(np.uint8))
