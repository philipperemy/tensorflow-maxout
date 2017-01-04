import numpy as np
import tensorflow as tf

"""

Maxout OP from https://arxiv.org/abs/1302.4389

Max pooling is performed in given filter/channel dimension. This can also be
used after fully-connected layers to reduce number of features.

Args:
    inputs: A Tensor on which maxout will be performed
    num_units: Specifies how many features will remain after max pooling at the
      channel dimension. This must be multiple of number of channels.
    axis: The dimension where max pooling will be performed. Default is the
      last dimension.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
Returns:
    A `Tensor` representing the results of the pooling operation.
Raises:
    ValueError: if num_units is not multiple of number of features.
"""


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


if __name__ == '__main__':
    with tf.Session() as sess:
        x = tf.Variable(np.random.uniform(size=(25, 10, 500)))
        y = tf.square(x)
        mo = max_out(x, 5, axis=2)
        sess.run(tf.global_variables_initializer())

        print(mo.eval().shape)
