
import tensorflow.python as tf
import tensorflow
from tensorflow import GradientTape
from tensorflow.python import keras
from tensorflow.python.keras import layers
from keras import losses
from keras import metrics
import numpy as np
import time
import sys
import os

from DataMaker import indexSpace_from_fake, normalize, vertexSpace_to_array
from MeshIO import read_spaces_from_directory, write_wavefront

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(devices[0], True)

print('AHHAHAHAHHAHAHAHAH',tensorflow.executing_eagerly())

BATCH_SIZE = 15
EPOCHS = 3
VERTEX_NULL_SPACE = -10.0
INDEX_NULL_SPACE = -1.0

VERTEX_COUNT = 8
DIMENSIONS = -1
INDEX_COUNT = 36



SHUFFLE_BUFFER_SIZE = 100
verts, inds = read_spaces_from_directory('./Processed_Spaces/', asTensor=True, seperate=True)
train_dataset = tf.data.Dataset.from_tensor_slices((verts,inds))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

class GAN(keras.Model):
    def __init__(self, vertex_latent_dims: int):
        super(GAN, self).__init__()
        self.vertex_latent_dimensions = vertex_latent_dims
        #Want to optomize the real losses, and deoptomize fake losses.
        #Want big values for reals, small for fake
        self.data_metrics = {'real index loss' : 0.0, 'fake index loss' : 0.0,
                       'real vertex loss' : 0.0, 'fake vertex loss' : 0.0,
                       'vertex accuracy' : metrics.BinaryCrossentropy(),
                       'index accuracy' : metrics.BinaryCrossentropy()}
        
        self.vertex_generator = keras.Sequential([
            keras.layers.Input(shape=(vertex_latent_dims,)),
            layers.Dense(1 * 1 * 1 * 128),
            layers.Reshape((1, 1, 1, 128)),
            layers.Conv3DTranspose(128, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv3D(3, kernel_size=4, padding='same', activation='sigmoid')
        ])
        
        self.index_generator = keras.Sequential([
            keras.layers.Input(shape=(3, 2, 2, 2)),
            layers.Conv3D(128, kernel_size=2, strides=1, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(20, activation='tanh'),
            layers.Dense(INDEX_COUNT, activation='sigmoid')
        ])
        
        
        self.vertex_discriminator = keras.Sequential([
            keras.layers.Input(shape=(3, 2, 2, 2)),
            layers.Conv3D(64, kernel_size=2, strides=1, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])
        self.index_discriminator = keras.Sequential([
            keras.layers.Input(shape=(2, 2, 2)),
            layers.Conv2D(64, kernel_size=2, strides=1, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])      
    
    def train_step(self, data):
        real_vertex_space, real_index_space = data
        random_latent_vectors = tensorflow.random.normal(shape=(BATCH_SIZE, self.vertex_latent_dimensions))

        with GradientTape() as vertex_tape:
            vertex_space = self.vertex_generator(random_latent_vectors)
            vertex_space = tensorflow.reshape(vertex_space, shape=(BATCH_SIZE, 3, 2, 2, 2))
            
        with GradientTape() as index_tape:
            index_array_proportions = self.index_generator(vertex_space)
          #  print(index_array_proportions)
            
            props = index_array_proportions.numpy()
            index_arrays = [[]] * BATCH_SIZE
            for a in range(BATCH_SIZE):
                for b in range(INDEX_COUNT):
                    value = props[a][b]
                    index_arrays[a].append(value * VERTEX_COUNT) 
        
        index_spaces = []
        for a in range(BATCH_SIZE):
            index_space = indexSpace_from_fake(vertex_space[a, :, :, :, :], index_arrays[a], INDEX_NULL_SPACE, VERTEX_NULL_SPACE, asTensor=False)
            normalize(index_space, null_space=INDEX_NULL_SPACE)
            index_space = tensorflow.convert_to_tensor(index_space)
            index_spaces.append(index_space)
        index_spaces = tensorflow.convert_to_tensor(index_spaces)
         
        labels = [
            tensorflow.ones((BATCH_SIZE, 1)),
            tensorflow.zeros((BATCH_SIZE, 1))
        ]
        labels[0] += 0.05 * tensorflow.random.uniform(tensorflow.shape(labels[0]))
        labels[1] += 0.05 * tensorflow.random.uniform(tensorflow.shape(labels[1]))   
        
        with GradientTape() as vertex_discrim_tape:
            loss_fake = self.vertex_discrim_loss_function(labels[1], self.vertex_discriminator(vertex_space))
            loss_real = self.vertex_discrim_loss_function(labels[0], self.vertex_discriminator(real_vertex_space))
            self.data_metrics['real vertex loss'] = loss_real
            loss = loss_fake + loss_real
            
        vertex_discrim_gradients = vertex_discrim_tape.gradient(loss, self.vertex_discriminator.trainable_weights)
        self.vertex_discrim_optomizer.apply_gradients(zip(vertex_discrim_gradients, self.vertex_discriminator.trainable_weights))
        
        with GradientTape() as index_discrim_tape:
            loss_fake = self.index_discrim_loss_function(labels[1], self.index_discriminator(index_spaces))
            loss_real = self.index_discrim_loss_function(labels[0], self.index_discriminator(real_index_space))
            self.data_metrics['real index loss'] = loss_real
            loss = loss_fake + loss_real
              
        index_discrim_gradients = index_discrim_tape.gradient(loss, self.index_discriminator.trainable_weights)
        self.index_discrim_optomizer.apply_gradients(zip(index_discrim_gradients, self.index_discriminator.trainable_weights))
            
        with GradientTape() as vertex_tape:
            vertex_space = self.vertex_generator(random_latent_vectors)
            vertex_space = tensorflow.reshape(vertex_space, shape=(BATCH_SIZE, 3, 2, 2, 2))
            
        with GradientTape() as index_tape:
            index_array_proportions = self.index_generator(vertex_space)
            
            props = index_array_proportions.numpy()
            index_arrays = [[]] * BATCH_SIZE
            for a in range(BATCH_SIZE):
                for b in range(INDEX_COUNT):
                    value = props[a][b]
                    index_arrays[a].append(value * VERTEX_COUNT) 
            
        index_spaces = []
        for a in range(BATCH_SIZE):
            index_space = indexSpace_from_fake(vertex_space[a, :, :, :, :], index_arrays[a], INDEX_NULL_SPACE, VERTEX_NULL_SPACE, asTensor=False)
            normalize(index_space, null_space=INDEX_NULL_SPACE)
            index_space = tensorflow.convert_to_tensor(index_space)
            index_spaces.append(index_space)
        index_spaces = tensorflow.convert_to_tensor(index_spaces)

        print('HAHAHHAHAHAHHA' * 100)
        vertex_output = self.vertex_discriminator(vertex_space)
        vertex_loss = self.vertex_discrim_loss_function(tensorflow.ones((BATCH_SIZE, 1)), vertex_output)
        self.data_metrics['fake vertex loss'] = vertex_loss
        
        index_output = self.index_discriminator(index_spaces)
        index_loss = self.index_discrim_loss_function(tensorflow.ones((BATCH_SIZE, 1)), index_output)
        self.data_metrics['fake index loss'] = index_loss
        vertex_loss += index_loss
        
        #Train the vertex generator on the fake loss of the index space, as the index generator is dependent on the vertex generator  
        vertex_generator_gradients = vertex_tape.gradient(vertex_loss, self.vertex_generator.trainable_weights)
        self.vertex_generator_optomizer.apply_gradients(zip(vertex_generator_gradients, self.vertex_generator.trainable_weights))
        
        index_generator_gradients = index_tape.gradient(index_loss, self.index_generator.trainable_weights)
        self.index_generator_optomizer.apply_gradients(zip(index_generator_gradients, self.index_generator.trainable_weights))
        
        self.data_metrics['vertex accuracy'].update_state(tensorflow.ones(BATCH_SIZE,1), vertex_output)
        self.data_metrics['index accuracy'].update_state(tensorflow.ones(BATCH_SIZE,1), index_output)
        #return self.data_metrics
        print('hello!')
    

    
        
    
    def set_params(self, discriminator_optomizer, generator_optomizer, discriminator_loss_function):
        self.index_discrim_loss_function = discriminator_loss_function
        self.vertex_discrim_loss_function =  discriminator_loss_function
        self.vertex_discrim_optomizer = discriminator_optomizer
        self.index_discrim_optomizer = discriminator_optomizer
        self.vertex_generator_optomizer = generator_optomizer
        self.index_generator_optomizer = generator_optomizer
        
    def call(self, inputs = None, training=None, mask=None):
        random_latent_vectors = tensorflow.random.normal(self.vertex_latent_dimensions)
        vertex_space = self.vertex_generator(random_latent_vectors)
        vertex_space = tensorflow.reshape(vertex_space, shape=(3, 2, 2, 2))
        vertex_array = vertexSpace_to_array(vertex_space, VERTEX_NULL_SPACE, isGenerated=True)
        index_array_proportions = self.index_generator(vertex_space).numpy()
        index_array = np.array([int(VERTEX_COUNT * a) for a in index_array_proportions])
        
        return vertex_array, index_array
    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
               self.train_step(image_batch)

            #Save the current output to a wavefront (.obj) file
            vertex_array, index_array = self.call()
            write_wavefront(f'./Model-Output/Validation1/epoch-{epoch}.obj', vertex_array, index_array)

            # Save the model every 15 epochs
            #if (epoch + 1) % 15 == 0:
            #checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
    def get_config(self):
        pass
    
    
Model = GAN(30)
Model.set_params(discriminator_optomizer=tensorflow.keras.optimizers.Adam(1e-4), generator_optomizer=tensorflow.keras.optimizers.Adam(1e-4),
              discriminator_loss_function=losses.BinaryCrossentropy())
Model.compile(run_eagerly=True)
#Model.train(train_dataset, epochs=EPOCHS)
Model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS)
Model.save('./Models/')
print('HAHAHHAHAHAHHAHAHHAHAHHA')