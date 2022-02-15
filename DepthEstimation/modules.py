from __future__ import division
import tensorflow as tf
import numpy as np
import math


def leaky_relu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)


####################################################################################################################################
# 2D convolution wrapper
####################################################################################################################################
def conv2d_leaky(
    input, kernel_shape, bias_shape, strides=1, relu=True, padding="SAME", dil=1
):
    # Conv2D
    weights = tf.compat.v1.get_variable(
        "weights",
        kernel_shape,
        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        dtype=tf.float32,
    )
    biases = tf.compat.v1.get_variable(
        "biases",
        bias_shape,
        initializer=tf.compat.v1.truncated_normal_initializer(),
        dtype=tf.float32,
    )
    output = tf.nn.conv2d(
        input=input,
        filters=weights,
        strides=[1, strides, strides, 1],
        padding=padding,
        dilations=[1, dil, dil, 1],
    )
    output = tf.nn.bias_add(output, biases)

    # ReLU (if required)
    if relu == False:
        return output

    output = leaky_relu(output, 0.2)
    return output


def deconv2d_leaky(
    input, kernel_shape, bias_shape, outputShape, strides=1, relu=True, padding="SAME"
):

    # Conv2D
    weights = tf.compat.v1.get_variable(
        "weights",
        kernel_shape,
        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
        dtype=tf.float32,
    )
    biases = tf.compat.v1.get_variable(
        "biases",
        bias_shape,
        initializer=tf.compat.v1.truncated_normal_initializer(),
        dtype=tf.float32,
    )
    output = tf.nn.conv2d_transpose(
        input,
        weights,
        output_shape=outputShape,
        strides=[1, strides, strides, 1],
        padding=padding,
    )
    output = tf.nn.bias_add(output, biases)

    # ReLU (if required)
    if relu == False:
        print("WARNING: reLU disabled")
    else:
        output = leaky_relu(output, 0.2)
    return output


####################################################################################################################################
# 2D convolution wrapper
####################################################################################################################################
def dilated_conv2d_leaky(
    input, kernel_shape, bias_shape, name, rate=1, relu=True, padding="SAME"
):
    with tf.compat.v1.variable_scope(name):
        # Conv2D
        weights = tf.compat.v1.get_variable(
            "weights", kernel_shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )
        biases = tf.compat.v1.get_variable(
            "biases", bias_shape, initializer=tf.compat.v1.truncated_normal_initializer()
        )
        output = tf.nn.atrous_conv2d(input, weights, rate=rate, padding=padding)
        output = tf.nn.bias_add(output, biases)

        if relu == False:
            print("WARNING: reLU disabled")
        else:
            output = leaky_relu(output, 0.2)
    return output


def bilinear_upsampling_by_deconvolution(src):
    shape = src.get_shape().as_list()
    h = shape[1] * 2
    w = shape[2] * 2
    return deconv2d_leaky(
        src, [2, 2, shape[3], shape[3]], shape[3], [shape[0], h, w, shape[3]], 2, True
    )


def bilinear_upsampling_by_convolution(src):
    with tf.compat.v1.variable_scope("bilinear_upsampling_by_convolution"):
        shape = src.get_shape().as_list()
        height = shape[1] * 2
        width = shape[2] * 2
        channels = shape[3]
        upsampled_src = tf.image.resize(src, [height, width])
        upsampled_src = conv2d_leaky(
            upsampled_src, [2, 2, channels, channels], [channels]
        )
        return upsampled_src
