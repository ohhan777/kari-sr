import cv2
from osgeo import gdal
import numpy as np
import glob


def open_geotiff(path):
    ds = gdal.Open(path)
    assert ds != None, f'{path} is not found'
    bands = []
    for i in range(ds.RasterCount):
        bands.append(ds.GetRasterBand(1).ReadAsArray().astype(np.uint16))
    img = np.dstack(bands) if ds.RasterCount > 1 else bands[0]
    return img

if __name__ == "__main__":
    files = glob.glob('../data/wv3-k3a/train/K3A/*.tif')
    for file in files:
        img = open_geotiff(file)
        img = (img/(2**14) - 1)*255.0
        out_path = file.replace('train/K3A','train/K3A_png').replace('.tif','.png')
        cv2.imwrite(out_path, img.astype(np.uint8))