import numpy as np
import tensorflow as tf


def read(path: str, vertex_count: int) -> np.ndarray:
    lines = open(path).readlines()
    vertices = np.array([(0.0,0.0,0.0)] * vertex_count, dtype=np.float32)
    
    a = 0
    for line in lines:
        contents = line.split(' ')
        if contents[0] == 'v':
            cords = (float(contents[1]), float(contents[2]), float(contents[3]))
            vertices[a] = cords
            a += 1

    return vertices

def to_box_undefined_dimensions(vertices: np.ndarray, error: tf.float32, augment: bool = False, ind_null_space:
           tf.float32 = -1.0,
           vert_null_space: tf.float32 = -1.0,
           debug: bool = True):
    # Indices of index map correlate to vertex indices in the vertex array
    box_index_map = [[0, 0, 0]] * vertices.size

    indices = [i for i in range(vertices.size)]
    x_s = np.array(list(map(lambda x: (x[0][0], x[1]), vertices, indices))).sort()
    y_s = np.array(map(lambda x: (x[0][1], x[1]), zip(vertices, indices))).sort()
    z_s = np.array(map(lambda x: (x[0][2], x[1]), zip(vertices, indices))).sort()
    del indices

    x_error_map = []
    y_error_map = []
    z_error_map = []

    xDim, yDim, zDim = 0, 0, 0
    xSuc, ySuc, zSuc = False, False, False
    failCount = 0
    for a in range(0, len(vertices) - 1):
        # X ERROR
        x_error = abs(x_s[a][0] - x_s[a + 1][0])
        x_error_map.append((a, x_error))
        if x_error >= error:
            xDim += 1
            xSuc = True

        # Y ERROR
        y_error = abs(y_s[a][0] - y_s[a + 1][0])
        y_error_map.append((a, y_error))
        if y_error >= error:
            yDim += 1
            ySuc = True

        # Z ERROR
        z_error = abs(z_s[a][0] - z_s[a + 1][0])
        z_error_map.append((a, z_error))
        if z_error >= error:
            zDim += 1
            zSuc = True

        if not (xSuc or ySuc or zSuc):
            if x_error >= y_error and z_error >= z_error:
                xDim += 1
            elif y_error >= x_error and y_error >= z_error:
                yDim += 1
            else:
                zDim += 1
            failCount = failCount + 1
        xSuc, ySuc, zSuc = False, False, False

        box_index_map[x_s[a][1]][0] = xDim
        box_index_map[y_s[a][1]][1] = yDim
        box_index_map[z_s[a][1]][2] = zDim

    box_space = tf.fill((xDim, yDim, zDim, 3), vert_null_space, dytpe=tf.float32)
    if augment:
        box_space.concat(tf.fill(box_space.shape[:-1]), ind_null_space, dtype=tf.float32)

    for b in range(0, len(box_index_map)):
        box_space[box_index_map[b] + [0]] = vertices[b][0]
        box_space[box_index_map[b] + [1]] = vertices[b][1]
        box_space[box_index_map[b] + [2]] = vertices[b][2]
        if augment:
            box_space[box_index_map[b] + [3]] = b

    if debug:
        print(f'FAILS: {failCount} for an error of {error}')

    del x_s
    del y_s
    del z_s
    return box_space


def to_box_fixed_dimensions_explicit_error(vertices: np.ndarray, required_shape: tuple[int, int, int], error: tf.float32 = None,
    augment: bool = False, ind_null_space: tf.float32 = -1.0, vert_null_space: tf.float32 = -1.0, debug: bool = True):

    # Indices of index map correlate to vertex indices in the vertex array
    box_index_map = [[0, 0, 0]] * vertices.size

    box_space = tf.fill(required_shape + (3,), vert_null_space, dytpe=tf.float32)
    if augment:
        box_space.concat(tf.fill(box_space.shape[:-1]), ind_null_space, dtype=tf.float32)

    indices = [i for i in range(vertices.size)]
    x_s = np.array(map(lambda x: (x[0][0], x[1]), zip(vertices, indices))).sort()
    y_s = np.array(map(lambda x: (x[0][1], x[1]), zip(vertices, indices))).sort()
    z_s = np.array(map(lambda x: (x[0][2], x[1]), zip(vertices, indices))).sort()
    del indices

    x_error_map = []
    y_error_map = []
    z_error_map = []

    xDim, yDim, zDim = 0, 0, 0
    xSuc, ySuc, zSuc = False, False, False
    failCount = 0
    for a in range(0, len(vertices) - 1):
        # X ERROR
        x_error = abs(x_s[a][0] - x_s[a + 1][0])
        x_error_map.append((a, x_error))
        if x_error >= error:
            xDim += 1
            xSuc = True

        # Y ERROR
        y_error = abs(y_s[a][0] - y_s[a + 1][0])
        y_error_map.append((a, y_error))
        if y_error >= error:
            yDim += 1
            ySuc = True

        # Z ERROR
        z_error = abs(z_s[a][0] - z_s[a + 1][0])
        z_error_map.append((a, z_error))
        if z_error >= error:
            zDim += 1
            zSuc = True

        if not (xSuc or ySuc or zSuc):
            if x_error >= y_error and z_error >= z_error:
                xDim += 1
            elif y_error >= x_error and y_error >= z_error:
                yDim += 1
            else:
                zDim += 1
            failCount = failCount + 1
        xSuc, ySuc, zSuc = False, False, False

        box_index_map[x_s[a][1]][0] = xDim
        box_index_map[y_s[a][1]][1] = yDim
        box_index_map[z_s[a][1]][2] = zDim

    # Sort the error maps
    x_error_map.sort(key=lambda x: x[1])
    y_error_map.sort(key=lambda x: x[1])
    z_error_map.sort(key=lambda x: x[1])

    def in_map(map_, curr_dim, req_dim, index):
        n = curr_dim - req_dim
        if n <= 0: return False
        for a in range(n):
            if map_[a][0] == index:
                return True
        return False

    lossX, lossY, lossZ = 0, 0, 0
    for b in range(0, len(box_index_map)):
        failed = False
        if in_map(x_error_map, xDim, required_shape[0], b):
            lossX = lossX + 1
            failed = True
        if in_map(y_error_map, yDim, required_shape[1], b):
            lossY = lossY + 1
            failed = True
        if in_map(z_error_map, zDim, required_shape[2], b):
            lossZ = lossZ + 1
            failed = True

        if failed:
            continue

        index = (box_index_map[b][0] - lossX, box_index_map[b][1] - lossY, box_index_map[b][2] - lossZ)
        box_space[index + (0,)] = vertices[b][0]
        box_space[index + (1,)] = vertices[b][1]
        box_space[index + (2,)] = vertices[b][2]
        if augment:
            box_space[index + (3,)] = b

    if debug:
        print(f'FAILS: {failCount} for an error of {error}')

    del x_s
    del y_s
    del z_s
    del x_error_map
    del y_error_map
    del z_error_map
    return box_space


def write_box(path: str, vertex_space, null_space: tf.float32):
    #Vertex space has no index aguement 
    f = open(path, 'w')
    f.write(f'V_Space_Dimensions: {vertex_space.shape[0]} {vertex_space.shape[1]} {vertex_space.shape[2]} {vertex_space.shape[3]}\n\n')
    
    i = 0
    three_size = vertex_space.shape[1] * vertex_space.shape[2] * vertex_space.shape[3]
    two_size = vertex_space.shape[2] * vertex_space.shape[3]
    while i < vertex_space.size:
        index = (
                  (i %  three_size) // two_size,
                  (i % two_size) // vertex_space.shape[3],
                  i % vertex_space.shape[3] ) 
        
        i += 1
        if vertex_space[(0,) + index] == null_space:
            continue
        
        vert = (vertex_space[(0,) + index], vertex_space[(1,) + index], vertex_space[(2,) + index])
        f.write(f'v {vert[0]} {vert[1]} {vert[2]} {index[0]} {index[1]} {index[2]}\n')

def read_box(path: str, null_space: tf.float32):
    lines = open(path).readlines()
    
    vertex_space_shape = lines[0].split(' ')
    vertex_space_shape = (int(vertex_space_shape[1]), int(vertex_space_shape[2]), int(vertex_space_shape[3]), int(vertex_space_shape[4]))
    
    vertex_space = np.empty(shape=vertex_space_shape, dtype=np.float32)
    vertex_space.fill(null_space)
    for line in lines[2:]:
        split = line.split(' ')
        
        if split[0] == 'v':
            index = (int(split[4]), int(split[5]), int(split[6]))
            vert = (float(split[1]), float(split[2]), float(split[3]))
            vertex_space[(0,) + index] = vert[0]
            vertex_space[(1,) + index] = vert[1]
            vertex_space[(2,) + index] = vert[2]
           
    return vertex_space