
from DataMaker import indexSpace_from_real, normalize, vertex_space
from MeshIO import read_wavefront, write_space, write_wavefront
import tensorflow.keras.utils as utils
import sys



verts, inds = read_wavefront('./Test/SHITCUBE.obj')
space = vertex_space(verts, desired_shape=(2,2,2), vertex_null_space=-2.0, index_augment=True, index_null_space=-1.0)
print(space, '\n')
normalize(space[:3, : ,: , :], -2.0)
##print(space)
index_space = indexSpace_from_real(space, inds, index_null_space=-1.0)
print(index_space, '\n')
normalize(index_space, -1.0)
print(index_space)
write_space('./Processed_Spaces/Cube2x2.txt', space[:3, : ,: , :], index_space, vertex_null_space=-2.0, index_null_space=-1.0)
#write_box('./Processed_Spaces/cube2x2.obj', space, index_space, vert_null_space=-10.0, ind_null_space=-1.0)

