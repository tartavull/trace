import tifffile
import h5py
import dataprovider.transform as transform
import click

@click.group()
def split_data():
    pass

@split_data.command()
def split():
    with tifffile.TiffFile('train-volume.tif') as tif:
        input = tif.asarray()
        with h5py.File('train-input.h5', 'w') as f:
            f.create_dataset('main', data = input[:27,:,:])
        with h5py.File('validation-input.h5', 'w') as f:
            f.create_dataset('main', data = input[27:,:,:])
    with tifffile.TiffFile('test-volume.tif') as tif:
        labels = tif.asarray()
        with h5py.File('train-labels.h5','w') as f:
            f.create_dataset('main', data = labels[:27,:,:])
        with h5py.File('validation-labels.h5','w') as f:
            f.create_dataset('main', data=labels[27:,:,:])

if __name__ == '__main__':
    split_data()
