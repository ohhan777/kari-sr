import numpy as np
from osgeo import gdal
import cv2
import glob

def open_geotiff(path):
    ds = gdal.Open(path)
    assert ds != None, f'{path} is not found'
    img = ds.GetRasterBand(1).ReadAsArray().astype(np.uint16)
    return img


path_list_k3a = glob.glob('/mnt/ohhan/storage_ssd/ai/Datasets/SR_exp/K3A/Shanghai/PAN/*.tif')
path_list_wv3 = glob.glob('/mnt/ohhan/storage_ssd/ai/Datasets/SR_exp/WV3/Shanghai/PAN/*.tif')

imgs = np.empty((1, 1300,1300), dtype=np.uint16)
for path in path_list_wv3:
    print(path)
    img = open_geotiff(path)
    imgs = np.append(imgs, np.expand_dims(img, 0), axis=0)
    if imgs.shape[0] >= 100:
        break
    
#img = open_geotiff('./data/WV3/SN3_roads_train_AOI_4_Shanghai_PAN_img119.tif')
# Normalize
#img = img / (2**14-1)  # K3A (14 bit-per-sample for PAN and MS)
#img = img / (2**11-1)  # WV-3 (11 bit-per-sample for PAN, 14 bit-per-sample for MS)
#cv2.imwrite('test.png', (img*255.0).astype(np.uint8))
imgs = imgs[1:]
print(imgs.mean(), imgs.min(), imgs.max())
print('Done')
pass