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


def to_box(vertices: np.ndarray, error: tf.float32, augment: bool = False, null_space: tf.float32 = -1.0,
           debug: bool = True):
    # Indices of index map correlate to vertex indices in the vertex array
    box_index_map = np.array(vertices.size, dtype=tf.int16)

    #TODO Look up if this makes a copy or not
    x_s = np.array(map(lambda x: x[0], vertices)).sort()
    y_s = np.array(map(lambda x: x[1], vertices)).sort()
    z_s = np.array(map(lambda x: x[2], vertices)).sort()

    xDim, yDim, zDim = 0, 0, 0
    xFail, yFail, zFail = False, False, False
    failCount = 0
    for a in range(0, len(vertices) - 1):
        # X ERROR
        if abs(x_s[a] - x_s[a + 1]) <= error:
            xDim = xDim + 1
            xFail = True
        # Y ERROR
        if abs(y_s[a] - y_s[a + 1]) <= error:
            yDim = yDim + 1
            yFail = True
        # Z ERROR
        if abs(z_s[a] - z_s[a + 1]) <= error:
            zDim = zDim + 1
            zFail = True

        if xFail or yFail or zFail:
            xDim = xDim + 1
            failCount = failCount + 1
        xFail, yFail, zFail = False, False, False

        np.append(box_index_map, (xDim, yDim, zDim))

    box_space = tf.constant(shape=(xDim, yDim, zDim, 3), dytpe=tf.float32)
    if augment:
        box_space.concat(tf.fill(shape=box_space.shape[:-1]), null_space, dtype=tf.float32)

    for b in range(0, len(box_index_map)):
        box_space[box_index_map[b] + (0,)] = vertices[b][0]
        box_space[box_index_map[b] + (1,)] = vertices[b][1]
        box_space[box_index_map[b] + (2,)] = vertices[b][2]
        if augment:
            box_space[box_index_map[b] + (3,)] = b

    if debug:
        print(f'FAILS: {failCount} for an error of {error}')

    del x_s
    del y_s
    del z_s
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
