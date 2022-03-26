
from DataMaker import adj_index_map_from_real, to_box_fixed_dimensions, to_reg_with_fake
from MeshIO import read, write


verts, inds = read('./Test/Cube.obj', )
print(inds)
print(verts, '\n\n')
space = to_box_fixed_dimensions(verts, required_shape=(2,2,2), vert_null_space=-29.0, augment=False, ind_null_space=-20.0)
print(space)
# index_space = adj_index_map_from_real(space[3, :, :, :], inds, -20.0)
# print(index_space)

