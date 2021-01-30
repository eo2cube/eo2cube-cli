import os
import zipfile as zf
import shutil
from pathlib2 import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tarfile

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)

def unzipper(item, output_dir):
    #print('Unzipping ' + str(item))
    if (item.endswith("tar.gz")):
        path = item.split('/')[-1]
        folder = path.split('.')[0]
        tar = tarfile.open(item)
        tar.extractall(output_dir + '/' + folder)
        tar.close()
    if (item.endswith("zip")):
        zip_ref = zf.ZipFile(item, "r")  # create zipfile object
        zip_ref.extractall(output_dir)  # extract file to dir
        zip_ref.close()



def sen2cor_single(sentinelfile, l2a_path):
    cmd = "{} {}".format(l2a_path, sentinelfile)
    os.system(cmd)

def jp2_to_tif(name, path, bands, clean):
    out = name.split(".")[0]
    if not os.path.exists(out):
        os.makedirs(out)
    for band in bands:
        for filename in Path(name).rglob("*"+ str(band) + ".jp2"):
            band_out = out + "/" + band + ".tif"
            cmd = "gdal_translate -of GTiff {} {}".format(filename, band_out)
            os.system(cmd)
    for mtd in Path(name).rglob("MTD_DS.xml"):
        shutil.copy(mtd, out + '/MTD.xml')
    if clean:
        print(name)
        shutil.rmtree(name)

def to_wgs84(file,out):

    dst_crs = 'EPSG:4326'

    with rasterio.open(file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(out + '/' + file.split('/')[-1], 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
