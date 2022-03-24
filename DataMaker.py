import numpy as np
import tensorflow as tf


#Puts the verticies in a box space with a predefined size
def to_box_fixed_dimensions(vertices: np.ndarray, required_shape: tuple[int, int, int], augment: bool = False,
           ind_null_space: tf.float32 = -1.0,  vert_null_space: tf.float32 = -1.0, debug: bool = True):

    # Indices of index map correlate to vertex indices in the vertex array
    box_index_map = np.array([[0, 0, 0]] * vertices.shape[0])

    box_space = np.zeros(shape=required_shape + (3,), dtype=np.float32)
    box_space.fill(vert_null_space)
    if augment:
        con = np.empty(shape=required_shape, dtype=np.float32)
        con.fill(ind_null_space)
        np.concatenate((box_space, con), axis=3)

    indices = [i for i in range(vertices.shape[0])]
    x_s = list(map(lambda x, y: (x[0], y), vertices, indices))
    y_s = list(map(lambda x, y: (x[1], y), vertices, indices))
    z_s = list(map(lambda x, y: (x[2], y), vertices, indices))
    x_s.sort(key=lambda x: x[0])
    y_s.sort(key=lambda x: x[0])
    z_s.sort(key=lambda x: x[0])
    x_s = np.array(x_s, dtype=np.float32)
    y_s = np.array(y_s, dtype=np.float32)
    z_s = np.array(z_s, dtype=np.float32)
    
    del indices

    (errX, errY, errZ), (x_err_map, y_err_map, z_err_map) = error_calcs(x_s, y_s, z_s)

    xDim, yDim, zDim = 0, 0, 0
   #! WARNING: LAST VERTICE NOT INCLUDED IN THIS ARRAY BECAUSE IT HAS NO ERROR VALUE
    for a in range(0, vertices.shape[0] - 1):
        # X ERROR
        if x_err_map[a][1] >= errX:
            xDim += 1

        # Y ERROR
        
        if y_err_map[a][1] >= errY:
            yDim += 1

        # Z ERROR

        if z_err_map[a][1] >= errZ:
            zDim += 1

        print(f'index {a} errz {errZ} and error {z_err_map[a][1]} Dimensions {zDim} value {z_s[a][0]}')
        print('YAYAYA' , x_s[a][1])
        box_index_map[int(x_s[a][1])][0] = xDim
        box_index_map[int(y_s[a][1])][1] = yDim
        box_index_map[int(z_s[a][1])][2] = zDim

   
    box_index_map[int(x_s[-1][1])][0] = xDim
    box_index_map[int(y_s[-1][1])][1] = yDim
    box_index_map[int(z_s[-1][1])][2] = zDim
    print(xDim, yDim, zDim)
    print([(a, b) for a, b in zip(box_index_map, vertices)], len(box_index_map))

    #Sort the error maps
    list(x_err_map).sort(key=lambda x: x[1])
    list(y_err_map).sort(key=lambda x: x[1])
    list(z_err_map).sort(key=lambda x: x[1])

    def in_map(map_, curr_dim, req_dim, index):
        n = curr_dim - req_dim
        if n <= 0: return False
        for a in range(n + 1):
            if map_[a][0] == index:
                return True
        return False

    print('DIM ', (xDim, yDim, zDim))
    lossX, lossY, lossZ = 0, 0, 0
    for b in range(0, len(box_index_map)):
        failed = False
        if in_map(x_err_map, xDim, required_shape[0], b):
            lossX = lossX + 1
            failed = True
        if in_map(y_err_map, yDim, required_shape[1], b):
            lossY = lossY + 1
            failed = True
        if in_map(z_err_map, zDim, required_shape[2], b):
            lossZ = lossZ + 1
            failed = True

        if failed:
            print('HAHHAHHAAHHA')
            continue

        index = (box_index_map[b][2] - lossZ, box_index_map[b][1] - lossY, box_index_map[b][0] - lossX)
        box_space[(0,) + index] = vertices[b][0]
        box_space[(1,) + index] = vertices[b][1]
        box_space[(2,) + index] = vertices[b][2]
        if augment:
            box_space[index + (3,)] = b

    del x_s
    del y_s
    del z_s
    del x_err_map
    del y_err_map
    del z_err_map
    return box_space

#Assumes the index of each element in the 3D matrix is convertible to 1D coordinates
def to_reg_with_fake(box_space, null_space: tf.float32) -> np.ndarray:
    assert (box_space.shape[3] == 4)
    vertices = np.array([], dtype=np.float32)

    for i in range(box_space.shape[0] * box_space.shape[1] * box_space.shape[2]):
        index = (
                  i // (box_space.shape[1] * box_space.shape[2]),
                  (i % (box_space.shape[1] * box_space.shape[2]) ) //box_space.shape[2],
                  i % box_space.shape[2] ) 
        
        if box_space[(0,) + index]  == null_space:
            continue
        
        vertices = np.append(vertices, (box_space[(0,) + index], box_space[(1,) + index], box_space[(2,) + index]))

    return vertices

#Using the augmented index map from the to_box func
def adj_index_map_from_real(adj_map_slice : np.ndarray, indices: np.ndarray, null_space: tf.float32) -> np.ndarray:
    index_map = list_to_dict(indices)
    adj_index_map = np.empty(shape=adj_map_slice.shape, dtype=np.float32)
    adj_index_map.fill(null_space)

    for i in range(0, adj_map_slice.size):
        index = (
                  i // (adj_map_slice.shape[1] * adj_map_slice.shape[2]),
                  (i % (adj_map_slice.shape[1] * adj_map_slice.shape[2]) ) // adj_map_slice.shape[2],
                  i % adj_map_slice.shape[2] ) 
        

        if adj_map_slice[index] == null_space:
            continue
        
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
def adj_index_map_fake(vertex_space: np.ndarray, indices: np.ndarray, ind_null_space: tf.float32, vert_null_space: tf.float32):
    index_map = list_to_dict(indices)
    adj_index_map = np.empty(shape=vertex_space[:-1], dtype=np.float32)
    adj_index_map.fill(ind_null_space)

    for i in indices:
        #Rows, Columns, Depth
        index = ((i % (adj_index_map.shape[0] * adj_index_map.shape[1])) // adj_index_map.shape[0],
                 i % adj_index_map.shape[1], i // (adj_index_map.shape[0] * adj_index_map.shape[1]))
        
        if vertex_space[(0,) + index] == vert_null_space:
            continue
        
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



def error_calcs(x_s: np.ndarray, y_s: np.ndarray, z_s: np.ndarray):
    errX, errY, errZ = 0.0, 0.0, 0.0
    errorsX = np.array([(0.0, 0.0)] * x_s.shape[0], dtype=np.float32)
    errorsY = np.array([(0.0, 0.0)] * x_s.shape[0], dtype=np.float32)
    errorsZ = np.array([(0.0, 0.0)] * x_s.shape[0], dtype=np.float32)
    print('X SORTED ', x_s)
    print('\nY SORTED', y_s)
    print('\nZ SORTED', z_s)
    for a in range(1, x_s.shape[0]):
        error = (abs(x_s[a][0] - x_s[a - 1][0]), abs(y_s[a][0] - y_s[a - 1][0]), abs(z_s[a][0] - z_s[a - 1][0]))
        errorsX[a] =  (a, error[0])
        errorsY[a] =  (a, error[2])
        errorsZ[a] =  (a, error[1])

        errX += error[0]
        errY += error[1]
        errZ += error[2]

    errX /= x_s.size - 1
    errY /= y_s.size - 1
    errZ /= z_s.size - 1
    return (errX, errY, errZ), (errorsX, errorsY, errorsZ)


