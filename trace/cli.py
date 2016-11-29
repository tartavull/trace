# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import webbrowser
import subprocess

import h5py

import click

import dataset_config
import trace
import models


def config_dict(x):
    return {
        'snemi3d': dataset_config.snemi3d_config(),
        'isbi': dataset_config.isbi_config()
    }[x]


def model_dict(x):
    return {
        'n4': models.default_N4()
    }[x]


@click.group()
def cli():
    pass


@cli.command()
def download():
    import dataset_config
    dataset_config.maybe_create_all_datasets(0.9)


@cli.command()
@click.argument('split', type=click.Choice(['train', 'validation', 'test']))
@click.argument('dataset', type=click.Choice(['snemi3d', 'isbi']))
@click.option('--aff/--no-aff', default=False, help="Display only the affinities.")
@click.option('--ip', default='172.17.0.2', help="IP address for serving")
@click.option('--port', default=4125, help="Port for serving")
def visualize(dataset, split, aff, ip, port):
    """
    Opens a tab in your webbrowser showing the chosen dataset
    """
    import neuroglancer

    config = config_dict(dataset)

    neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer(voxel_size=[6, 6, 30])
    if aff:
        import augmentation
        augmentation.maybe_create_affinities(split)
        add_affinities(config.folder, split + '-affinities', viewer)
    else:
        add_file(config.folder, split + '-input', viewer)
        add_file(config.folder, split + '-labels', viewer)

    print('open your brower at:')
    print(viewer.__str__().replace('172.17.0.2', '54.89.114.140')) # Replace the second argument with your own server's ip address
    webbrowser.open(viewer.__str__())
    print("press any key to exit")
    input()


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
@click.argument('split', type=click.Choice(['train', 'validation', 'test']))
@click.argument('dataset', type=click.Choice(['snemi3d', 'isbi']))
@click.option('--high', type=float, default=0.9)
@click.option('--low', type=float, default=0.3)
@click.option('--dust', type=int, default=250)
def watershed(dataset, split, high, low, dust):
    """
    TODO Explain what each argument is, dust is currently ignored
    """

    config = config_dict(dataset)

    curent_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(["julia",
                     curent_dir +"/thirdparty/watershed/watershed.jl",
                     config.folder + split + "-affinities.h5",
                     config.folder + split + "-labels.h5",
                     str(high),
                     str(low)])


@cli.command()
@click.argument('model_type', type=click.Choice(['n4']))
@click.argument('dataset', type=click.Choice(['snemi3d', 'isbi']))
def train(model_type, dataset):
    """
    Train an N4 models to predict affinities
    """

    trace.train(model_dict(model_type), config_dict(dataset))


@cli.command()
@click.argument('model_type', type=click.Choice(['n4']))
@click.argument('dataset', type=click.Choice(['snemi3d', 'isbi']))
@click.argument('split', type=click.Choice(['train', 'validation', 'test']))
def predict(model_type, dataset, split):
    """
    Realods a model previously trained
    """
    import trace
    trace.predict(model_dict(model_type), config_dict(dataset), split)


@cli.command()
def grid():
    trace.grid_search()

if __name__ == '__main__':
    cli()
