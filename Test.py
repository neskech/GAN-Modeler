
from Debug import read, read_box, write_box
from MeshIO import to_box_fixed_dimensions


verts = read('./Test/SHITCUBE.obj', vertex_count=19)
#print(verts)
new = to_box_fixed_dimensions(verts, (3,3,3), augment=False, vert_null_space=-29.0)
write_box('./Test/SHIT_SPACE.obj', new, null_space=-29.0)
new = read_box('./Test/SHIT_SPACE.obj', null_space=-29.0)
print(new, new.shape)

