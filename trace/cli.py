# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import webbrowser
import subprocess

import h5py

import numpy as np
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
def convertresult():
    snemi3d.convert_result()

@cli.command()
@click.argument('dataset', type=click.Choice(['train', 'test']))
@click.option('--aff/--no-aff', default=False, help="Display only the affinities.")
@click.option('--ip', default='172.17.0.2', help="IP address for serving")
@click.option('--port', default=4125, help="Port for serving")
def visualize(dataset, aff, ip, port):
    """
    Opens a tab in your webbrowser showing the chosen dataset
    """
    import neuroglancer

    snemi3d_dir = snemi3d.folder()
    neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    neuroglancer.set_server_bind_address(bind_address='172.17.0.2', bind_port=4125)
    viewer = neuroglancer.Viewer(voxel_size=[6, 6, 30])
    if aff:
        import augmentation
        augmentation.maybe_create_affinities(dataset)
        add_affinities(snemi3d_dir, dataset+'-affinities', viewer)
    else:
        add_file(snemi3d_dir, dataset+'-input', viewer)
        add_file(snemi3d_dir, dataset+'-labels', viewer)

    print('open your brower at:')
    print(viewer.__str__().replace('172.17.0.2', '52.53.186.131')) # Replace the second argument with your own server's ip address
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



@cli.command()
@click.argument('dataset', type=click.Choice(['train', 'test']), default='test')
@click.option('--high', type=float, default=0.9)
@click.option('--low', type=float, default=0.3)
@click.option('--dust', type=int, default=250)
def watershed(dataset, high, low, dust):
    """
    TODO Explain what each argument is, dust is currently ignored
    """
    curent_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     curent_dir+"/thirdparty/watershed/watershed.jl",
                     snemi3d.folder()+dataset+"-affinities.h5",
                     snemi3d.folder()+dataset+"-labels.h5",
                     str(high),
                     str(low)])

@cli.command()
def train():
    """
    Train an N4 models to predict affinities
    """
    import trace
    trace.train()


@cli.command()
def predict():
    """
    Realods a model previously trained
    """
    import trace
    trace.predict()

if __name__ == '__main__':
    cli()
