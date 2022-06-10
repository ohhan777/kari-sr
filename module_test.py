import numpy as np
from osgeo import gdal

def open_geotiff(path):
    ds = gdal.Open(path)
    band = []
    for i in range(4):
        band.append(ds.GetRasterBand(i + 1).ReadArray())
    img = np.dstack(band).astype(np.int16)
    return img

img = open_geotiff('./data/K3A/BLD01190_PAN_K3A_NIA0373.tif')
pass