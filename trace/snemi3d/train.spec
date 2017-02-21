[files]
img = /home/trace-local/tests/../trace/snemi3d/train-input.h5
lbl = /home/trace-local/tests/../trace/snemi3d/train-labels.h5

[params]
max_trans = [23.0]*20

[image]
file = img
preprocess = dict(type='rescale')

[label]
file = lbl
transform = dict(type='affinitize')

[dataset]
input = image
label = label

