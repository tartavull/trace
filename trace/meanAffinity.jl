using Agglomeration
using Process
using HDF5

aff_path = ARGS[1]
seg_path = ARGS[2]
ground_truth_path = ARGS[3]
print("aff input path: ")
print(aff_path)
print("\nseg input path: ")
print(seg_path)
print("\nground truth input path: ")
print(ground_truth_path)

dend_path = ARGS[4]
dendValues_path = ARGS[5]
rand_path = ARGS[6]
print("\ndend output path: ")
print(dend_path)
print("\ndendValues output path: ")
print(dendValues_path)
print("\nrand output path: ")
print(rand_path)

aff = h5read(aff_path, "main")
seg = h5read(seg_path, "main")
#ground_truth = h5read(ground_truth_path, "main")

#dend, dendValues, randError = Process.forward(aff, seg, human_labels=ground_truth)
dend, dendValues = Process.forward(aff, seg)

#print("\n rand shape:")
#print(size(randError))
h5write(dend_path, "main", dend)
h5write(dendValues_path, "main", dendValues)
#h5write(rand_path, "main", randError)

