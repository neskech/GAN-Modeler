import numpy as np



def read_wavefront(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Reads a wavefront (.obj) file

    Args:
        path (str): path to the file

    Returns:
        tuple[np.ndarray, np.ndarray]: Vertex and index arrays
    """
    
    lines = open(path).readlines()
    vertices = []
    indices = []

    for line in lines:
        contents = line.split(' ')
        
        if contents[0] == 'v':
            cords = (float(contents[1]), float(contents[2]), float(contents[3]))
            vertices.append(cords)
            
        elif contents[0] == 'f':        
            #Remove any '/'s slashes
            if contents[1].find('/') != -1:
                for a in range(len(1, contents)):
                    contents[a] = contents[a][:contents[a].find('/')]
            
            #If only three indices are present, they are easy to work with
            if len(contents) == 4:   
                ind = (int(contents[1]) - 1, int(contents[2]) - 1, int(contents[3]) - 1)
                indices.append(ind[0])
                indices.append(ind[1])
                indices.append(ind[2])
                
            #If four indices are present, they make up a 'face'. We must make that face into two triangles
            else:
                ind = (int(contents[1]) - 1, int(contents[2]) - 1, int(contents[3]) - 1, int(contents[4]) - 1)
                
                indices.append(ind[0])
                indices.append(ind[1])
                indices.append(ind[3])
                
                indices.append(ind[1])
                indices.append(ind[2])
                indices.append(ind[3])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int16)


def write_wavefront(path: str, vertices: np.ndarray, indices: np.ndarray):
    """Writes to a wavefront (.obj) file in a given directory

    Args:
        path (str): Desired path
        vertices (np.ndarray): Vertex array
        indices (np.ndarray): index array
    """
    f = open(path, 'w')
    for vert in vertices:
        f.write(f'v {vert[0]} {vert[1]} {vert[2]} \n')

    for a in range(0, indices.size - 2, 3):
        f.write(f'f {indices[a]} {indices[a + 1]} {indices[a + 2]} \n')


def read_space(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Reads a vertex and index space from a text (.txt) file

    Args:
        path (str): Desired path

    Returns:
        tuple[np.ndarray, np.ndarray]: vertex and index space arrays
    """
    lines = open(path).readlines()
    
    vertex_space_shape = lines[0].split(' ')
    vertex_space_shape = (int(vertex_space_shape[1]), int(vertex_space_shape[2]), int(vertex_space_shape[3]), int(vertex_space_shape[4]))
    index_space_shape = lines[2].split(' ')
    index_space_shape = (int(index_space_shape[1]), int(index_space_shape[2]), int(index_space_shape[3]))
    
    vert_null_space = float(lines[1].split(' ')[1])
    ind_null_space = float(lines[3].split(' ')[1])
    
    vertex_space = np.empty(shape=vertex_space_shape, dtype=np.float32)
    vertex_space.fill(vert_null_space)
    index_space = np.empty(shape=index_space_shape, dtype=np.float32)
    index_space.fill(ind_null_space)
    
    for line in lines[5:]:
        split = line.split(' ')
        
        if split[0] == 'v':
            index = (int(split[4]), int(split[5]), int(split[6]))
            vert = (float(split[1]), float(split[2]), float(split[3]))
            vertex_space[(0,) + index] = vert[0]
            vertex_space[(1,) + index] = vert[1]
            vertex_space[(2,) + index] = vert[2]
        else:
            index = (int(split[2]), int(split[3]), int(split[4]))
            el = float(split[1])
            index_space[index] = el
           
    return vertex_space, index_space 
        
def write_space(path: str, vertex_space : np.ndarray, index_space : np.ndarray, vertex_null_space: np.float32, index_null_space: np.float32):
    """Writes a vertex and index space to a text (.txt) file

    Args:
        path (str): Desired path
        vertex_space (np.ndarray): Vertex space array
        index_space (np.ndarray): Index space array
        vert_null_space (np.float32): Vertex null space number
        ind_null_space (np.float32): Index null space number
    """
    f = open(path, 'w')
    
    f.write(f'V_Space_Dimensions: {vertex_space.shape[0]} {vertex_space.shape[1]} {vertex_space.shape[2]} {vertex_space.shape[3]}\n')
    f.write(f'Vert_Space_Null_Space: {vertex_null_space}\n')
    f.write(f'I_Space_Dimensions: {index_space.shape[0]} {index_space.shape[1]} {index_space.shape[2]}\n')
    f.write(f'Index_Space_Null_Space: {index_null_space}\n\n')
    

    threeD_size = vertex_space.shape[1] * vertex_space.shape[2] * vertex_space.shape[3]
    twoD_size = vertex_space.shape[2] * vertex_space.shape[3]
    #Write the vertex space
    for i in range(vertex_space.size):
        index = (     #1D -> 3D index conversion
                  (i %  threeD_size) // twoD_size,
                  (i % twoD_size) // vertex_space.shape[3],
                  i % vertex_space.shape[3] ) 
        
        if vertex_space[(0,) + index] == vertex_null_space:
            continue
        
        vert = (vertex_space[(0,) + index], vertex_space[(1,) + index], vertex_space[(2,) + index])
        f.write(f'v {vert[0]} {vert[1]} {vert[2]} {index[0]} {index[1]} {index[2]}\n')
       
    #Write the index space 
    for i in range(index_space.size):
        index = (       #1D -> 3D index conversion
                (i %  threeD_size) // twoD_size, 
                (i % twoD_size) // vertex_space.shape[3],
                i % vertex_space.shape[3] ) 
        
        if index_space[index] == index_null_space:
            continue

        f.write(f'f {index_space[index]} {index[0]} {index[1]} {index[2]}\n')
        

