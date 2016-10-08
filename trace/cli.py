# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import h5py
import webbrowser
import neuroglancer

import click

import snemi3d

@click.group()
def cli():
    pass

@cli.command()
def download():
    import snemi3d
    snemi3d.maybe_create_dataset()

@cli.command()
@click.argument('dataset')
def visualize(dataset):
    snemi3d_dir = snemi3d.folder()

    neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
    viewer = neuroglancer.Viewer(voxel_size=[6, 6, 30])
    if dataset == 'train':
        add_file(snemi3d_dir, 'train-input.h5', viewer)
        add_file(snemi3d_dir, 'train-labels.h5', viewer)
    elif dataset == 'test':
        add_file(snemi3d_dir, 'test-input.h5', viewer)
        add_file(snemi3d_dir, 'test-labels.h5', viewer)
    else:
        raise ValueError('Only options available are test or train')

    print('open your brower at:')
    print(viewer.__str__())
    webbrowser.open(viewer.__str__())
    print("press any key to exit")
    raw_input()

def add_file(folder, filename, viewer):
    try:
        with h5py.File(folder+filename,'r') as f:
            arr = f['main'][:]
            viewer.add(arr, name=filename)
    except IOError:
        print(filename+' not found')
if __name__ == '__main__':
    cli()