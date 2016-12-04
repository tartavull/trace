[files]
img = test-volume.h5
lbl = test-labels.h5

[image]
file = img
preprocess = dict(type='rescale')

[label]
file = lbl
transform = dict(type='affinitize')

[dataset]
input = image
label = label
