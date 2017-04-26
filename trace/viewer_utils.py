import h5py


def add_file(folder, filename, viewer):
    print(folder + filename + '.h5')
    try:
        with h5py.File(folder+filename+'.h5', 'r') as f:
            arr = f['main'][:]
            viewer.add(arr, name=filename)
    except IOError:
        print(filename+' not found')

def add_raw(folder, filename, viewer):
    try:
        with h5py.File(folder+filename+'.h5', 'r') as f:
            arr = f['volumes']['raw'][:]
            viewer.add(arr, name=filename)
    except IOError:
        print(filename+' not found')

def add_labels(folder, filename, viewer):
    try:
        with h5py.File(folder+filename+'.h5', 'r') as f:
            print(folder)
            if "clefts" in folder:
              print('yes')
              arr = f['volumes']['labels']['clefts'][:]
            else:
              arr = f['volumes']['labels']['neuron_ids'][:]
            viewer.add(arr, name=filename + '_labels')
    except IOError:
        print(filename+' not found')


def add_affinities(folder, filename, viewer):
    """
    This is holding all the affinities in RAM,
    it would be easy to modify so that it is
    reading from disk directly.
    """
    try:
        with h5py.File(folder+filename+'.hdf', 'r') as f:
            x_aff = f['main'][0, :, :, :]
            viewer.add(x_aff, name=filename+'-x', shader="""
            void main() {
              emitRGB(
                    vec3(1.0 - toNormalized(getDataValue(0)),
                         0,
                         0)
                      );
            }
            """)
            y_aff = f['main'][1, :, :, :]
            viewer.add(y_aff, name=filename+'-y', shader="""
            void main() {
              emitRGB(
                    vec3(0,
                         1.0 - toNormalized(getDataValue(0)),
                         0)
                      );
            }
            """)
            z_aff = f['main'][2, :, :, :]
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
