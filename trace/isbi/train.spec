[files]
img = /Users/maxgoldstein/opt/seung/trace/trace/isbi/train-input.h5
lbl = /Users/maxgoldstein/opt/seung/trace/trace/isbi/train-labels.h5

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

