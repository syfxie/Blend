import tensorflow as tf
import numpy as np
from tensorflow import linalg
from tensorflow import expand_dims
from tensorflow import shape
from tensorflow import cast

def gram_matrix(tensor):
    '''determines the strength of the correlation between two style features'''
    temp = tensor
    print("calculating gram matrix: ")
    
    # get shape of input tensor
    shape = tf.shape(temp)
    print("original shape: ")
    print(shape)

    # flatten tensor into 2D
    t_reshape = tf.reshape(temp, [shape[2], shape[1] * shape[0]])
    print("flattened tensor: ")
    print(tf.shape(t_reshape))

    # multiply matrix by its transpose
    result = tf.matmul(t_reshape, t_reshape, transpose_b=True)
    gram_matrix = tf.expand_dims(result, axis=0)
    print("gram matrix: ")
    return gram_matrix