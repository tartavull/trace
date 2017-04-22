using Agglomeration
using Process
using HDF5

function load_segmentation(fn)
	 return h5read(fn, "main")
end

aff = load_segmentation("cremi/a/results/unet_3d_4layers/run-maxpoolz_2d_conv_half0.9995_15k/test-pred-affinities.h5")
#segm = load_segmentation("validation-labels.h5")
segm = load_segmentation("tmp-validation-labels-new-gaussian.h5")

#aff is a (X,Y,Z,3) array of affinities
#segm is a (X,Y,Z) array containing a flat segmentation from watershed
#dend, dendValues = Process.forward(aff, segm)
#dend will be a (2,n) array of pairs of segment ids from segm
#dendValues will be a (n,) array of values ranging from 0.0 to 1.0
#Restricting dend to the pairs corresponding with entries in
#dendValues above threshold t gives a minimum spanning tree
#of the result of mean affinity agglomeration a threshold t


#dend, dendValues = Process.forward(aff, segm)

merge_tree = Process.forward2(aff, segm)

#seg = MergeTrees.flatten(segm, merge_tree, 0.5)
#h5write("mean_affinity_segm0.5.h5", "main", seg)

create a flat segmentation at threshold 0.5
for i = 0:9
    seg = MergeTrees.flatten(segm, merge_tree, i*0.1)
    thres = round(i*0.1,1)
    h5write("mean_affinity_segm$(thres).h5", "main", seg)
end