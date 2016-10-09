# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import webbrowser

import h5py
import neuroglancer
import numpy as np
import click

import snemi3d
import augmentation

@click.group()
def cli():
    pass

@cli.command()
def download():
    import snemi3d
    snemi3d.maybe_create_dataset()

@cli.command()
@click.argument('dataset', type=click.Choice(['train', 'test']))
@click.option('--aff/--no-aff', default=False, help="Display only the affinities.")
def visualize(dataset, aff):
    """
    Opens a tab in your webbrowser showing the chosen dataset
    """
    snemi3d_dir = snemi3d.folder()
    neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
    viewer = neuroglancer.Viewer(voxel_size=[6, 6, 30])
    if aff:
        augmentation.maybe_create_affinities(dataset)
        add_affinities(snemi3d_dir, dataset+'-affinities', viewer)
    else:
        add_file(snemi3d_dir, dataset+'-input', viewer)
        add_file(snemi3d_dir, dataset+'-labels', viewer)
    
    print('open your brower at:')
    print(viewer.__str__())
    webbrowser.open(viewer.__str__())
    print("press any key to exit")
    raw_input()

def add_file(folder, filename, viewer):
    try:
        with h5py.File(folder+filename+'.h5','r') as f:
            arr = f['main'][:]
            viewer.add(arr, name=filename)
    except IOError:
        print(filename+' not found')

def add_affinities(folder, filename, viewer):
    """
    This is holding all the affinities in RAM,
    it would be easy to modify so that it is
    reading from disk directly.
    """
    try:
        with h5py.File(folder+filename+'.h5','r') as f:
            x_aff = f['main'][0,:,:,:]
            viewer.add(x_aff, name=filename+'-x', shader="""
            void main() {
              emitRGB(
                    vec3(1.0 - toNormalized(getDataValue(0)),
                         0,
                         0)
                      );
            }
            """)
            y_aff = f['main'][1,:,:,:]
            viewer.add(y_aff, name=filename+'-y', shader="""
            void main() {
              emitRGB(
                    vec3(0,
                         1.0 - toNormalized(getDataValue(0)),
                         0)
                      );
            }
            """)
            z_aff = f['main'][2,:,:,:]
            viewer.add(z_aff, name=filename+'-z', shader="""
            void main() {
              emitRGB(
                    vec3(0,
                         0,
                         1.0 - toNormalized(getDataValue(0)))
                      );
            }
            """)
    except IOError:
        print(filename+'.h5 not found')

if __name__ == '__main__':
    cli()