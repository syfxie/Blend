import tensorflow as tf
# calculate algebraic summation in the Gram matrix formuls
from tensorflow import linalg

# add dimension to tensor
from tensorflow import expand_dims

# shape of tensor
from tensorflow import shape


from tensorflow import cast


def gram_matrix(input_t):
    '''returns the multivariant correlation two filters in a cnn block'''
    # implement the Graham matrix formula 

    # calculate summation (numerator)
    result = linalg.einsum('bijc, bijd->bcd', input_t, input_t)
    # expand dimensions to add outer batch dimension as the first dimension of the result tensor
    numerator = expand_dims(result, 0)

    # calculate IJ product (denominator)
    # get shape of the input tensor
    shape = tf.shape(input_t)
    # convert shape into a float for IJ value
    denom = tf.cast(shape[1], shape[2], dtype=tf.float32)

    # calculate gram matrix
    gram_matrix = numerator / denom
    return gram_matrix




   

