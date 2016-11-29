"""
Download and decompress SNEMI3D
"""

from __future__ import print_function
import os.path
import urllib
import zipfile
import h5py
from tifffile import TiffFile

def maybe_download(base_url, folder, filename):
  full_url = base_url + filename
  full_path = folder + filename
  if not os.path.exists(full_path):
    print("downloading "+full_url)
    urllib.urlretrieve (full_url, full_path)

def maybe_create_hdf5(folder, filename):
  full_path = folder + filename
  path_without_ext , ext = os.path.splitext(full_path)

  if not os.path.exists(path_without_ext+'.tif'):
    zip_ref = zipfile.ZipFile(full_path, 'r')
    zip_ref.extractall(folder)
    zip_ref.close()

  if not os.path.exists(path_without_ext+'.h5'):
    with TiffFile(path_without_ext+'.tif') as tif:
      arr = tif.asarray() 
      with h5py.File(path_without_ext+'.h5') as f:
        print('created '+path_without_ext+'.h5')
        f.create_dataset('main',data=arr)

def maybe_create_dataset():
  snemi3d_dir = folder()
  if not os.path.exists(snemi3d_dir):
    os.mkdir(snemi3d_dir)
    
  base_url = "http://brainiac2.mit.edu/SNEMI3D/sites/default/files/"
  maybe_download(base_url, snemi3d_dir, "train-volume.tif")
  maybe_download(base_url, snemi3d_dir, "train-labels.tif")
  maybe_download(base_url, snemi3d_dir, "test-volume.tif")
  maybe_create_hdf5(snemi3d_dir, "train-volume.zip")
  maybe_create_hdf5(snemi3d_dir, "train-labels.zip")
  maybe_create_hdf5(snemi3d_dir, "train-volume.zip")

def folder():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  snemi3d_dir = current_dir + '/snemi3d/'
  return snemi3d_dir

if __name__ == '__main__':
  maybe_create_dataset()