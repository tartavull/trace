from __future__ import print_function
import h5py
import os.path

from thirdparty.DataProvider.python.data_provider import VolumeDataProvider

def maybe_create_affinities(dataset):
    """
    Args:
        dataset (str): either train or test

        It should recompute them in the case that the labels changes
    """
    if os.path.exists('snemi3d/'+dataset+'-affinities.h5'):
        print ('Affinties already exists.')
        return

    if not os.path.exists('snemi3d/'+dataset+'-labels.h5'):
        print ('No labels from where to compute affinities.')
        return
    #there is a little bug in DataProvider that doesn't let us do it
    #for the actual size of the dataset 100,1024,1024
    patch_size = (99,1023,1023)
    net_spec = {'label':patch_size}
    params = {'augment': [] , 'drange':[0]}
    spec = dataset+'-snemi3d.spec'
    dp = VolumeDataProvider(spec, net_spec, params)
    affinities =  dp.random_sample()['label']
    with h5py.File('snemi3d/'+dataset+'-affinities.h5','w') as f:
        f.create_dataset('main',data=affinities)