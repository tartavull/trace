===============================
Trace
===============================


.. image:: https://img.shields.io/pypi/v/trace.svg
        :target: https://pypi.python.org/pypi/trace

.. image:: https://img.shields.io/travis/tartavull/trace.svg
        :target: https://travis-ci.org/tartavull/trace


Map your brain with Deep Learning

Blog post https://medium.com/p/map-your-brain-with-deep-learning-17737d36bf80

* Free software: MIT license
* Documentation: https://tartavull.github.io/trace


Features
--------

* TODO



===========================
TIPS
===========================

Setting up Neuroglancer
-----------------------

1) Change the second argument of neuroglancer.set_server_bind_address to your
server's ip address.

2) Unblock inbound port 4125 on your server.


Troubleshooting Docker
----------------------

- Docker tends to crash a lot when doing training. You can run tensorboard
in a screen session, but make sure to run the actual training in the default
session, not in screen.

- When you stop and restart a GPU EC2 instance, you have to restart nvidia-docker
  using the command:

    sudo -b nohup nvidia-docker-plugin > /tmp/nvidia-docker.log

    - DO NOT RUN THIS MORE THAN ONCE

    - If your docker is still not starting, and you already committed an image
      and uploaded it to dockerhub you can restart your docker entirely by
      doing this:
      (THIS WILL DELETE EVERYTHING IN YOUR DOCKER SO MAKE YOU COMMITTED A 
      DOCKER IMAGE AND UPLOADED IT TO DOCKERHUB)
        
        sudo service docker stop
        sudo rm -rf /var/lib/docker
        sudo rm -rf /usr/local/lib/nvidia-docker
        sudo rm -rf /usr/local/lib/nvidia-docker-plugin
        sudo service docker start

        Reinstalling ndivida-docker plugin
        Redownloading the image

            - (I would recommend committing a docker image every so often,
              to make sure you don't lose your results)


- Stop docker containers before exiting an instance, to make sure they stop
  properly and don't give you trouble when starting them enxt time.


How to create a training/validation split
-----------------------------------------

Go into the folder with the training input and labels, and start the python
interactive shell.

If the data is in tif format:
::
    import tifffile
    import h5py
    import dataprovider.transform as transform

    with tifffile.TiffFile('[name of your training input].tif') as tif:
        input = tif.asarray()

    with h5py.File('training-input.h5', 'w') as f:
        # I chose to have a 90-10 split with 27 of the 30 layers of ISBI data
        # as training data, but feel free to use your own split.
        f.create_dataset('main', data=input[:27,:,:])
    with h5py.File('validation-input.h5', 'w') as f:
        f.create_dataset('main', data=input[27:,:,:])

    with tifffile.TiffFile('[name of your training label].tif') as tif:
        labels = tif.asarray()

    with h5py.File('training-labels.h5', 'w') as f:
        f.create_dataset('main', data=labels[:27,:,:])
    with h5py.File('validation-labels.h5', 'w') as f:
        f.create_dataset('main', data=labels[27:,:,:])

    aff = transform.affinitize(labels)
    with h5py.File('training-affinities.h5', 'w') as f:
        f.create_dataset('main', data=labels[:,:27,:,:])
    with h5py.File('validation-affinities.h5', 'w') as f:
        f.create_dataset('main', data=labels[:,27:,:,:])

If the data is in h5 format, everything is the same, except the input/label
files can be read in this way instead
::
    import h5py
    with h5py.File('[name of your training input].h5', 'r') as f:
        input = f['main']

    
Layer Activation/Weight Visualisation
-------------------------------------

TODO
