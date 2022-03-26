
from DataMaker import adj_index_map_from_real, to_box_fixed_dimensions, to_reg_with_fake
from MeshIO import read, read_box, write, write_box
import sys


vert, ind = read_box('./Test/Space.txt', )
print(vert)
print(ind)
sys.exit()


verts, inds = read('./Test/SHITCUBE.obj', )
print(verts, '\n\n')
space = to_box_fixed_dimensions(verts, required_shape=(2,2,2), vert_null_space=-29.0, augment=True, ind_null_space=-20.0)
print(space)
print(inds)
index_space = adj_index_map_from_real(space[3, :, :, :], inds, -20.0)
write_box('./Test/Space.txt', space, index_space, vert_null_space=-29.0, ind_null_space=-20.0)

