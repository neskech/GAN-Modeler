import tensorflow as tf
from typing import List
import numpy as np

#! REMEMBER TO DELETE THE TENSORS RETURNED BY THESE FUNCTIONS TO PREVENT MEMORY LEAKS


def read(path: str, face_count: int) -> tuple[np.ndarray, np.ndarray]:
    lines = open(path).readlines()
    vertex_count = face_count * 3 - (face_count - 1) * 2
    index_count = face_count * 3
    vertices = np.array(vertex_count, dtype=(tf.float32, tf.float32, tf.float32))
    indices = np.array(index_count, dtype=tf.int16)

    for line in lines:
        contents = line.split(" ")
        if contents[0] == 'v':
            cords = (float(contents[1]), float(contents[2]), float(contents[3]))
            np.append(vertices, cords)
        else:
            ind = (int(contents[1]), int(contents[2]), int(contents[2]))
            np.append(indices, ind)

    return vertices, indices


def write(path: str, vertices: np.ndarray, indices: np.ndarray):
    f = open(path)
    for vert in vertices:
        f.write(f'v {vert[0]} {vert[1]} {vert[2]} \n')

    for a in range(0, indices.size - 3, 3):
        f.write(f'f {indices[a]} {indices[a + 1]} {indices[a + 2]} \n')


def to_box(vertices: np.ndarray, error: tf.float32, augment: bool = False, ind_null_space: tf.float32 = -1.0,
           vert_null_space: tf.float32 = -1.0,
           debug: bool = True):
    # Indices of index map correlate to vertex indices in the vertex array
    box_index_map = [[0, 0, 0]] * vertices.size

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


#Puts the verticies in a box space with a predefined size
def to_box_good(vertices: np.ndarray, required_shape: tuple[int, int, int], error: tf.float32, augment: bool = False,
           ind_null_space: tf.float32 = -1.0,  vert_null_space: tf.float32 = -1.0, debug: bool = True):

    # Indices of index map correlate to vertex indices in the vertex array
    box_index_map = [[0, 0, 0]] * vertices.size

    box_space = tf.fill(required_shape + (3,), vert_null_space, dytpe=tf.float32)
    if augment:
        box_space.concat(tf.fill(box_space.shape[:-1]), ind_null_space, dtype=tf.float32)

    #TODO Look up if this makes a copy or not
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
    #TODO start the loop at index 1. Add the first box index before the loop
    #TODO Make the sorted arrays tuples with the vertex index to whom the x,y,z cord belongs to. Use the 'a' index to index into sorted, then sorted into a box index array thats preallocated
    #TODO Insert the corresponding dimension (for example taking the index from x_s will have me inserting xDim into the correct index)
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
            #TODO Find the axis with the greatest error and move in the direction of that
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

    #Sort the error maps
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


def to_reg_with_aug(box_space, face_count: int) -> np.ndarray:
    assert (box_space.shape[3] == 4)
    vertex_count = face_count * 3 - (face_count - 1) * 2
    vertices = np.empty(vertex_count, dtype=(tf.float32, tf.float32, tf.float32))

    for i in range(box_space.shape[0] * box_space.shape[1] * box_space.shape[2]):
        index = ((i % (box_space.shape[0] * box_space.shape[1])) // box_space.shape[0],
                 i % box_space.shape[1], i // (box_space.shape[0] * box_space.shape[1]))

        vert_index = box_space[index + (3,)]
        if vert_index != -1:
            vertices[vert_index] = (box_space[index + (0,)], box_space[index + (1,)], box_space[index + (2,)])

    return vertices


#Assumes the index of each element in the 3D matrix is convertible to 1D coordinates
def to_reg_with_fake(box_space, vertex_count: int) -> np.ndarray:
    assert (box_space.shape[3] == 4)
    vertices = np.empty(vertex_count, dtype=(np.float32, np.float32, np.float32))

    for i in range(box_space.shape[0] * box_space.shape[1] * box_space.shape[2]):
        index = ((i % (box_space.shape[0] * box_space.shape[1])) // box_space.shape[0],
                 i % box_space.shape[1], i // (box_space.shape[0] * box_space.shape[1]))

        vertices[i] = (box_space[index + (0,)], box_space[index + (1,)], box_space[index + (2,)])

    return vertices

#Using the augmented index map from the to_box func
def adj_index_map_real(adj_map_slice, indices: np.ndarray, null_space: tf.float32 = -1.0):
    index_map = list_to_dict(indices)
    adj_index_map = tf.fill(adj_map_slice.shape, null_space, dtype=tf.float32)

    for i in range(0, adj_map_slice.size):
        index = ((i % (adj_map_slice.shape[0] * adj_map_slice.shape[1])) //adj_map_slice.shape[0],
                 i % adj_map_slice.shape[1], i // (adj_map_slice.shape[0] * adj_map_slice.shape[1]))

        sum = 0.0
        vals = (-1, 0, 1)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    direc = (vals[a], vals[b], vals[c])
                    new_index = (index[0] + direc[0], index[1] + direc[1], index[2] + direc[2])

                    #If the new index is in bounds and in the index hashmap and not a null space
                    if (0 <= new_index[0] < adj_map_slice.shape[0] and
                            0 <= new_index[1] < adj_map_slice.shape[1] and
                            0 <= new_index[2] < adj_map_slice[2] and
                            adj_map_slice[new_index] != null_space and
                            adj_map_slice[new_index] in index_map[adj_map_slice[index]]):
                        sum += .33 * direc[0] + .66 * direc[1] + 1.0 * direc[2]
        sum /= 3.0
        adj_index_map[index] = sum

    return adj_index_map


#Uses the 3D to 1D coordinate conversion rather than using an existing index map from the to_box func
def adj_index_map_fake(shape: tuple[int, int, int], indices: np.ndarray, null_space: tf.float32 = -1.0):
    index_map = list_to_dict(indices)
    adj_index_map = tf.fill(shape, null_space, dtype=tf.float32)

    for i in indices:
        #Rows, Columns, Depth
        index = ((i % (adj_index_map.shape[0] * adj_index_map.shape[1])) // adj_index_map.shape[0],
                 i % adj_index_map.shape[1], i // (adj_index_map.shape[0] * adj_index_map.shape[1]))

        sum = 0.0
        vals = (-1, 0, 1)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    direc = (vals[a], vals[b], vals[c])
                    new_index = (index[0] + direc[0], index[1] + direc[1], index[2] + direc[2])
                    oneD_index = new_index[0] * adj_index_map.shape[0] + new_index[1] + new_index[2] * adj_index_map.shape[0] * adj_index_map.shape[1]

                    #If the new index is in bounds and in the index hashmap and not a null space
                    if (0 <= new_index[0] < adj_index_map.shape[0] and
                            0 <= new_index[1] < adj_index_map.shape[1] and
                            0 <= new_index[2] < adj_index_map[2] and
                            oneD_index in index_map[i]):
                        sum += .33 * direc[0] + .66 * direc[1] + 1.0 * direc[2]
        sum /= 3
        adj_index_map[index] = sum

    return adj_index_map


def list_to_dict(list_items: np.ndarray) -> dict:
    hashMap = dict()
    for a in range(0, len(list_items), 3):

        if not hashMap.__contains__(list_items[a]):
            #add it
            hashMap[list_items[a]] = []

        if not list_items[a + 1] in hashMap[list_items[a]]: hashMap[list_items[a]].append(list_items[a + 1])
        if not list_items[a + 2] in hashMap[list_items[a]]: hashMap[list_items[a]].append(list_items[a + 2])

        if not hashMap.__contains__(list[a + 1]):
            # add it
            hashMap[list_items[a + 1]] = []

        if not list_items[a] in hashMap[list_items[a + 1]]: hashMap[list_items[a + 1]].append(list_items[a])
        if not list_items[a + 2] in hashMap[list_items[a + 1]]: hashMap[list_items[a + 1]].append(list_items[a + 2])

        if not hashMap.__contains__(list[a + 2]):
            # add it
            hashMap[list_items[a + 2]] = []

        if not list_items[a] in hashMap[list_items[a + 2]]: hashMap[list_items[a + 2]].append(list_items[a])
        if not list_items[a + 1] in hashMap[list_items[a + 2]]: hashMap[list_items[a + 2]].append(list_items[a + 1])

    return hashMap


def normalize(verticies: List[(float, float, float)], indices: List[int]):
    pass
