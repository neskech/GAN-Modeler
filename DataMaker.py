import numpy as np
from pyrsistent import v

def vertex_space(vertices: np.ndarray, desired_shape: tuple[int, int, int], index_augment: bool,
           vertex_null_space: np.float32, index_null_space: np.float32, debug: bool = False) -> np.ndarray:
    """Creates an four dimensional adjacency matrix from a list of vertices.
       Contains three 3D channels, of which represent the x, y, and z component of each vertex

    Args:
        vertices (np.ndarray): Vertex array
        desired_shape (tuple[int, int, int]): Desired shape for the space
        index_augment (bool, optional): Add an extra channel for vertex indices
        vertex_null_space (np.float32, optional): Null space number for the vertex space
        index_null_space (np.float32, optional): Null space number for the index space. -1.0 reccomended
        debug (bool, optional): Helpful debugging information. Defaults to False.
        
    Returns: (np.ndarray): The vertex adjacency matrix
    """


    #Create the vertex space, and concatenate it with an extra index channel if specified
    vertex_space = np.zeros(shape= (3,) + desired_shape, dtype=np.float32)
    vertex_space.fill(vertex_null_space)
    
    if index_augment:
        con = np.empty(shape= (1,) + desired_shape, dtype=np.float32)
        con.fill(index_null_space)
        vertex_space = np.concatenate((vertex_space, con), axis=0)

    #Create sorted component arrays for each axis. Each entry has a vertex array index to reference where it came from
    indices = [i for i in range(vertices.shape[0])]
    x_sorted = list(map(lambda x, y: (x[0], y), vertices, indices))
    y_sorted = list(map(lambda x, y: (x[1], y), vertices, indices))
    z_sorted = list(map(lambda x, y: (x[2], y), vertices, indices))
    
    #Sort according to the x, y, z values
    x_sorted.sort(key=lambda x: x[0])
    y_sorted.sort(key=lambda x: x[0])
    z_sorted.sort(key=lambda x: x[0])
    
    #Convert into numpy arrays
    x_sorted = np.array(x_sorted, dtype=np.float32)
    y_sorted = np.array(y_sorted, dtype=np.float32)
    z_sorted = np.array(z_sorted, dtype=np.float32)

    #Calculate the average errors and construct error maps for each component
    (errX, errY, errZ), (x_err_map, y_err_map, z_err_map) = error_calcs(x_sorted, y_sorted, z_sorted)

    #The index map stores the 3D index in the vertex space corresponding to each vertex
    #Each entry in the index map directly corresponds to an entry in the vertex array. 
    # I.E. index 0 of index map corresponds to index 0 of vertex array
    index_map = np.array([[0, 0, 0]] * vertices.shape[0])
    
    #The current dimensions of each axis. This dynamically increase as the vertex space
    #Is built up
    xDim, yDim, zDim = 0, 0, 0
    
    #Since the error of the first sorted components is not considered, the point must be manually added in
    index_map[int(x_sorted[0][1])][0] = xDim
    index_map[int(y_sorted[0][1])][1] = yDim
    index_map[int(z_sorted[0][1])][2] = zDim

    for a in range(1, vertices.shape[0]):
        # X ERROR
        if x_err_map[a - 1][0] >= errX:
            xDim += 1

        # Y ERROR
        if y_err_map[a - 1][0] >= errY:
            yDim += 1

        # Z ERROR
        if z_err_map[a - 1][0] >= errZ:
            zDim += 1

        #Set the indices in the index map. X, Y, and Z sorted all can be referencing different points at once 
        index_map[int(x_sorted[a][1])][0] = xDim
        index_map[int(y_sorted[a][1])][1] = yDim
        index_map[int(z_sorted[a][1])][2] = zDim

    #Sort the error maps
    list(x_err_map).sort(key=lambda x: x[1])
    list(y_err_map).sort(key=lambda x: x[1])
    list(z_err_map).sort(key=lambda x: x[1])


    lossX, lossY, lossZ = 0, 0, 0
    inds = []
    
    #If any of the dimensions go ount of bounds, cull the points with the least error
    #The points with the least error contrbute the least to the geometry, so they should be the first to go
    #The least error points will always be in the beginning of the error maps since they're now sorted
    for a in range(xDim + 1 - desired_shape[0]):
        inds.append(x_err_map[a][1])
        lossX += 1     
          
    for a in range(yDim + 1 - desired_shape[1]):
         if y_err_map[a][0] not in inds: 
             inds.append(y_err_map[a][1])
         lossY += 1
         
    for a in range(zDim + 1 - desired_shape[2]):
         if z_err_map[a][0] not in inds: 
             inds.append(z_err_map[a][1])
         lossZ += 1
     
    #Make an array of the remaining indices with a list comprehension     
    indices = [i for i in range(len(index_map)) if i not in inds]    
    
    #And add the appropiate vertices to the vertex space
    for b in indices:   
        index = (index_map[b][2] - lossZ, index_map[b][1] - lossY, index_map[b][0] - lossX)
        
        vertex_space[(0,) + index] = vertices[b][0]
        vertex_space[(1,) + index] = vertices[b][1]
        vertex_space[(2,) + index] = vertices[b][2]
        if index_augment:
            vertex_space[(3,) + index] = b

    return vertex_space


def vertexSpace_to_array(vertex_space, vertex_null_space: np.float32) -> np.ndarray:
    """Converts a vertex space with an index augment back to a regular vertex array

    Args:
        vertex_space (_type_): The vertex space
        vertex_null_space (np.float32): The null space number for the vertex array

    Returns:
        np.ndarray: The vertex array
    """
    
    #The vertex space must have an index augment for this function to perform properly
    assert (vertex_space.shape[0] == 4)
    vertices = []

    for i in range(vertex_space.shape[1] * vertex_space.shape[2] * vertex_space.shape[3]):
        index = (               #1D -> 3D index conversion
                  i // (vertex_space.shape[1] * vertex_space.shape[2]),
                  (i % (vertex_space.shape[1] * vertex_space.shape[2]) ) //vertex_space.shape[2],
                  i % vertex_space.shape[2] ) 
        
        #If we find a null space, simply skip over it
        if vertex_space[(0,) + index]  == vertex_null_space:
            continue
        
        augment_index = vertex_space[(3,) + index]
        #Associate the agument index with each vertex so we can refer back to the correct sorted order
        vertices.append((augment_index, (vertex_space[(0,) + index], vertex_space[(1,) + index], vertex_space[(2,) + index])))

    #Sort with respect to the augment index and return the raw components
    vertices.sort(key=lambda x: x[0])
    return np.array([vert[1] for vert in vertices])


def indexSpace_from_real(vertex_space: np.ndarray, indices: np.ndarray, index_null_space: np.float32) -> np.ndarray:
    """Constructs an index space from a 'real' 3D model

    Args:
        vertex_space (np.ndarray): The Vertex space
        indices (np.ndarray): The index array
        index_null_space (np.float32): Null space number for the index space

    Returns:
        np.ndarray: The index space
    """
    #The vertex space must have an index augment for this function to perform properly
    assert (vertex_space.shape[0] == 4)
    
    #A hashmap which maps indices to other indices which they are connected to
    index_map = list_to_dictionary(indices)
    index_space = np.empty(shape=vertex_space.shape, dtype=np.float32)
    index_space.fill(index_null_space)

    for i in range(vertex_space.size):
        index = (
                  i // (vertex_space.shape[1] * vertex_space.shape[2]),
                  (i % (vertex_space.shape[1] * vertex_space.shape[2]) ) // vertex_space.shape[2],
                  i % vertex_space.shape[2] ) 
    
        if vertex_space[(3,) + index] == index_null_space:
            continue
        
        sum = 0.0
        vals = (-1, 0, 1)
        
        #Iterate over all possible directions in 3D space from a point
        #The value, sum, is built off of whether or not adjacent places are conncted to the current 3D index
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    direc = (vals[a], vals[b], vals[c])
                    new_index = (index[0] + direc[0], index[1] + direc[1], index[2] + direc[2])

                    #If the new index is in bounds and in the index hashmap and not a null space
                    if (0 <= new_index[0] < vertex_space.shape[0] and
                            0 <= new_index[1] < vertex_space.shape[1] and
                            0 <= new_index[2] < vertex_space.shape[2] and
                            vertex_space[new_index] != index_null_space and
                            int(vertex_space[new_index]) in index_map[int(vertex_space[index])]):
                        sum += .33 * direc[0] + .66 * direc[1] - 1.0 * direc[2]
        sum /= 3.0
        index_space[index] = sum

    return index_space


def indexSpace_from_fake(vertex_space: np.ndarray, indices: np.ndarray, index_null_space: np.float32, vertex_null_space: np.float32) -> np.ndarray:
    """Creates an index space from a 3D model generated by the nerual network

    Args:
        vertex_space (np.ndarray): The vertex space
        indices (np.ndarray): The index array
        index_null_space (np.float32): The null space number for the index space
        vertex_null_space (np.float32): The null space number for the vertex space

    Returns:
        np.ndarray: The index space
    """
    #The index hashmap
    index_map = list_to_dictionary(indices)
    
    index_space = np.empty(shape=vertex_space[:-1], dtype=np.float32)
    index_space.fill(index_null_space)

    vert_index = 0
    null_indices = []
    for i in range(index_space.size):
        #Rows, Columns, Depth
        index = (
                  i // (index_space.shape[1] * index_space.shape[2]),
                  (i % (index_space.shape[1] * index_space.shape[2]) ) // index_space.shape[2],
                  i % index_space.shape[2] )     
        
        if vertex_space[(0,) + index] == vertex_null_space:
            null_indices.append(i)
            continue
        
        sum = 0.0
        vals = (-1, 0, 1)
        
        #Iterate over all possible directions in 3D space from a point
        #The value, sum, is built off of whether or not adjacent places are conncted to the current 3D index
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    direc = (vals[a], vals[b], vals[c])
                    new_index = (index[0] + direc[0], index[1] + direc[1], index[2] + direc[2])
                    oneD_index = new_index[0] * index_space.shape[1] * index_space.shape[2] + new_index[1] * index_space.shape[2] + new_index[2]
                    
                    
                    #Calculate what we need to subtract the new one dimensional index by to help it remain valid
                    offset = 0
                    while offset < len(null_indices) and oneD_index < null_indices[offset]:
                        offset += 1
                    oneD_index -= offset

                    #If the new index is in bounds and in the index hashmap and not a null space
                    if (0 <= new_index[0] < index_space.shape[0] and
                            0 <= new_index[1] < index_space.shape[1] and
                            0 <= new_index[2] < index_space[2] and
                            oneD_index in index_map[vert_index]):
                        sum += .33 * direc[0] + .66 * direc[1] - 1.0 * direc[2]
        sum /= 3
        index_space[index] = sum
        vert_index += 1

    return index_space


def list_to_dictionary(list_items: np.ndarray) -> dict:
    """Creates a dicionary (hashmap) where each index is a key to a list of values,
    of which are other indices it is connected with

    Args:
        list_items (np.ndarray): The list of indices

    Returns:
        dict: A hashmap of the index connections
    """
    hashMap = dict()
    for a in range(0, len(list_items) - 2, 3):
        #Iterates through the index array in 3's, as each triangle is made of three indices. 
        
        if not hashMap.__contains__(list_items[a]):
            #add it
            hashMap[list_items[a]] = []

        if not list_items[a + 1] in hashMap[list_items[a]]: hashMap[list_items[a]].append(list_items[a + 1])
        if not list_items[a + 2] in hashMap[list_items[a]]: hashMap[list_items[a]].append(list_items[a + 2])

        if not hashMap.__contains__(list_items[a + 1]):
            # add it
            hashMap[list_items[a + 1]] = []

        if not list_items[a] in hashMap[list_items[a + 1]]: hashMap[list_items[a + 1]].append(list_items[a])
        if not list_items[a + 2] in hashMap[list_items[a + 1]]: hashMap[list_items[a + 1]].append(list_items[a + 2])

        if not hashMap.__contains__(list_items[a + 2]):
            # add it
            hashMap[list_items[a + 2]] = []

        if not list_items[a] in hashMap[list_items[a + 2]]: hashMap[list_items[a + 2]].append(list_items[a])
        if not list_items[a + 1] in hashMap[list_items[a + 2]]: hashMap[list_items[a + 2]].append(list_items[a + 1])

    return hashMap



def error_calcs(x_s: np.ndarray, y_s: np.ndarray, 
    z_s: np.ndarray) -> tuple[tuple[np.float32, np.float32, np.float32], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
    """Calculates the average error over each axis. Caches calculations into three arrays

    Args:
        x_s (np.ndarray): Sorted X components
        y_s (np.ndarray): Sorted Y components
        z_s (np.ndarray): Sorted Z components

    Returns:
        tuple[np.float32, np.float32, np.float32]: A tuple of the average errors for the X, Y, and Z axes
        tuple[np.ndarray, np.ndarray, np.ndarray]: Cached error arrays for each axis. Combined with an index 
        for the corresponding
    """
    
    #The error for every point except for the first of each sorted array is taken into account
    errX, errY, errZ = 0.0, 0.0, 0.0
    errorsX = np.array([(0.0, 0.0)] * (x_s.shape[0] - 1), dtype=np.float32)
    errorsY = np.array([(0.0, 0.0)] * (x_s.shape[0] - 1), dtype=np.float32)
    errorsZ = np.array([(0.0, 0.0)] * (x_s.shape[0] - 1), dtype=np.float32)

    for a in range(1, x_s.shape[0]):
        error = (abs(x_s[a][0] - x_s[a - 1][0]), abs(y_s[a][0] - y_s[a - 1][0]), abs(z_s[a][0] - z_s[a - 1][0]))
        
        #Keep a tuple to reference the index of the vertex we're finding the error for
        errorsX[a - 1] =  (error[0], x_s[a][1])
        errorsY[a - 1] =  (error[2], y_s[a][1])
        errorsZ[a - 1] =  (error[1], z_s[a][1])

        errX += error[0]
        errY += error[1]
        errZ += error[2]

    errX /= x_s.size - 1
    errY /= y_s.size - 1
    errZ /= z_s.size - 1
    return (errX, errY, errZ), (errorsX, errorsY, errorsZ)


