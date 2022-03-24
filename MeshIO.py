from re import L
from pyrsistent import v
import tensorflow as tf
from typing import List
import numpy as np

#! REMEMBER TO DELETE THE TENSORS RETURNED BY THESE FUNCTIONS TO PREVENT MEMORY LEAKS


def read(path: str, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    lines = open(path).readlines()
    vertices = np.array([(0.0,0.0,0.0)] * vertex_count, dtype=np.float32)
    indices = np.array([], dtype=np.int16)

    a = 0
    for line in lines:
        contents = line.split(' ')
        if contents[0] == 'v':
            cords = (float(contents[1]), float(contents[2]), float(contents[3]))
            vertices[a] = cords
            a += 1
        else:
            ind = (int(contents[1]), int(contents[2]), int(contents[2]))
            indices = np.append(indices, ind[0])
            indices = np.append(indices, ind[1])
            indices = np.append(indices, ind[2])

    return vertices, indices


def write_obj(path: str, vertices: np.ndarray, indices: np.ndarray):
    f = open(path, 'w')
    for vert in vertices:
        f.write(f'v {vert[0]} {vert[1]} {vert[2]} \n')

    for a in range(0, indices.size - 3, 3):
        f.write(f'f {indices[a]} {indices[a + 1]} {indices[a + 2]} \n')


def read_box(path: str, null_space: tf.float32) -> tuple(np.ndarray, np.ndarray):
    lines = open(path).readlines()
    
    vertex_space_shape = lines[0].split(' ')
    vertex_space_shape = (int(vertex_space_shape[0]), int(vertex_space_shape[1]), int(vertex_space_shape[2]), int(vertex_space_shape[3]))
    index_space_shape = lines[1].split(' ')
    index_space_shape = (int(index_space_shape[0]), int(index_space_shape[1]), int(index_space_shape[2]))
    
    vertex_space = np.empty(shape=vertex_space_shape, dtype=np.float32)
    vertex_space.fill(null_space)
    index_space = np.empty(shape=index_space_shape, dtype=np.float32)
    index_space.fill(null_space)
    
    for line in lines[2:]:
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
        
def write_box(path: str, vertex_space : np.ndarray, index_space : np.ndarray, null_space: tf.float32):
    #Vertex space has no index aguement 
    f = open(path, 'w')
    f.write(f'V_Space_Dimensions: {vertex_space.shape[0]} {vertex_space.shape[1]} {vertex_space.shape[2]} {vertex_space.shape[3]}\n')
    f.write(f'I_Space_Dimensions: {index_space.shape[0]} {index_space.shape[1]} {index_space.shape[2]}\n')
    
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
        
        vert = (vertex_space[index + (0,)], vertex_space[index + (1,)], vertex_space[index + (2,)])
        f.write(f'v {vert[0]} {vert[1]} {vert[2]} {index[0]} {index[1]} {index[2]}\n')
        
    for i in range(index_space.size):
        index = ( (i %  three_size) // two_size,
                (i % two_size) // vertex_space.shape[3],
                i % vertex_space.shape[3] ) 

        f.write(f'f {index_space[index]} {index[0]} {index[1]} {index[2]}\n')
        

