import math, copy, json, subprocess, shutil, time, pickle, random, itertools, os, sys
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from functools import partial
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

auto = tf.data.AUTOTUNE
device_name = tf.test.gpu_device_name()

if device_name == '':
    print('Warning: No GPU!')

from tensorflow.keras import (
    models, optimizers, callbacks, Input, Model, activations, Sequential
    )
from tensorflow.keras.layers import (
    GRUCell, Dense, BatchNormalization, Flatten, Layer, Reshape, RNN,
    Normalization, LayerNormalization, BatchNormalization, SimpleRNNCell,
    SimpleRNN, LSTM, Flatten, GaussianNoise
    )
import tensorflow_probability as tfp

# making plots beautiful:

SMALL_SIZE = 20//1.4
MEDIUM_SIZE = 24//1.4
BIGGER_SIZE = 28//1.4

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)


plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


@tf.function
def yeo_johnson_transform(x, lambdas):
    condition_pos = tf.greater_equal(x, 0)
    result_pos = tf.where(
        lambdas != 0,
        (tf.pow(x + 1, lambdas) - 1) / lambdas,
        tf.math.log(x + 1)
        )
    result_neg = tf.where(
        lambdas != 2,
        -(tf.pow(-x + 1, 2 - lambdas) - 1) / (2 - lambdas),
        -tf.math.log(-x + 1)
        )

    return tf.where(tf.greater_equal(x, 0), result_pos, result_neg)

@tf.function
def z_score_normalize(normalizable_data, means, stds, lambda_dict=None, epsilon=1e-8):
    """
    Normalize data by subtracting the mean and dividing by the standard deviation (z-score normalization).

    Args:
        normalizable_data (dict): A dictionary of data arrays to be normalized.
                                  Each key is a data label, and each value is a tensor of data.
        means (dict): A dictionary of mean values for each data array.
        stds (dict): A dictionary of standard deviation values for each data array.
        epsilon (float): A small constant added to the standard deviation to prevent division by zero.

    Returns:
        dict: A dictionary of normalized data.
    """
    normalized_data = {}

    for key in normalizable_data:

        mean_tensor = tf.reshape(means[key], [1, 1, -1]) # for broadcasting
        std_tensor = tf.reshape(stds[key], [1, 1, -1]) # for broadcasting

        normalized_data[key] = tf.math.divide(
            tf.math.subtract(normalizable_data[key], mean_tensor),
            std_tensor + epsilon
        )

    return normalized_data

@tf.function
def transform(normalizable_data, keys=['node_features', 'targets'], mins=None, maxs=None, lambda_dict=None, cbrt=None, epsilon=1e-8):
    """
    Normalize data using min-max scaling to [-1, 1].
    """
    scaled_data = {}

    for key in keys:
        scaled_data[key] = tf.identity(normalizable_data[key])

        if lambda_dict[key] is not None:
            scaled_data[key] = yeo_johnson_transform(scaled_data[key], lambda_dict[key])

        if cbrt[key] is not None:
            scaled_data[key] = tf.experimental.numpy.cbrt(scaled_data[key])

        if mins[key] is not None and maxs[key] is not None:
            # for broadcasting
            min_tensor = tf.reshape(mins[key], [1, 1, -1])
            max_tensor = tf.reshape(maxs[key], [1, 1, -1])
            # min-max normalization
            scaled_data[key] = 2 * ((scaled_data[key] - min_tensor) / (max_tensor - min_tensor + epsilon)) - 1

    return scaled_data

def get_lambdas(ds_normalized, num_iterations, keys = ['node_features', 'targets'], alpha = 0.9):

    batch_lambdas, running_lambdas = {}, {}
    n = 0

    for sample in tqdm(ds_normalized.take(num_iterations)):
        n += 1
        for k in keys:
            data = sample[k]

            # Generate random indices for each of of first dim
            first_dim = data.shape[0]
            second_dim = data.shape[1]
            indices = tf.random.uniform(
                [first_dim], minval=0, maxval=second_dim, dtype=tf.int32
                )
            range_tensor = tf.range(first_dim)

            # Get indices of shape [first_dim, num_frames]
            gather_indices = tf.stack([range_tensor, indices], axis=1)

            data = tf.identity(tf.gather_nd(data, gather_indices))
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            pt.fit(data)
            batch_lambdas[k] = tf.constant(pt.lambdas_, dtype=tf.float32)

            if n == 1:
                running_lambdas[k] = batch_lambdas[k]
            else:
                prev_lambda = running_lambdas[k]
                running_lambdas[k] = alpha * prev_lambda + (1 - alpha) * batch_lambdas[k]

    return running_lambdas

def get_z_score_stats(ds_normalized, keys = ['node_features', 'edge_features', 'targets'], alpha = 0.9):

    eps = 10e-8
    n , running_mean, running_var, batch_mean, batch_var = 0, {}, {}, {}, {}

    for sample in tqdm(ds_normalized.take(10)):
        n += 1

        for k in keys:
            batch_mean[k], batch_var[k] = tf.nn.moments(sample[k], axes=0)
            if n == 1:
                running_mean[k] = batch_mean[k]
                running_var[k] = batch_var[k]
            else:
                running_mean[k] = alpha*running_mean[k]+(1-alpha)*batch_mean[k]
                running_var[k] = alpha*running_var[k]+(1-alpha)*batch_var[k]

    return {
        'means': running_mean,
        'stds': {k: tf.math.sqrt(running_var[k] + eps) for k in keys}
        }

def get_min_max_stats(ds_normalized, num_iterations, lambda_dict=None, cbrt=False, keys = ['node_features', 'targets'], alpha = 0.9):

    n , running_min, running_max, batch_min, batch_max = 0, {}, {}, {}, {}

    for sample in tqdm(ds_normalized.take(num_iterations)):
        n += 1
        x = {}
        for k in keys:
            data = sample[k]
            # Generate random indices for each of of first dim
            first_dim = data.shape[0]
            second_dim = data.shape[1]
            indices = tf.random.uniform(
                [first_dim], minval=0, maxval=second_dim, dtype=tf.int32
                )
            range_tensor = tf.range(first_dim)
            # Get indices of shape [first_dim, num_frames]
            gather_indices = tf.stack([range_tensor, indices], axis=1)
            data = tf.identity(tf.gather_nd(data, gather_indices))

            if lambda_dict != None:
                data = yeo_johnson_transform(data, lambda_dict[k])

            if cbrt == True:
                data = tf.experimental.numpy.cbrt(data)

            batch_min[k], batch_max[k] = tf.reduce_min(data, axis=0), tf.reduce_max(data, axis=0)


    return {
        'mins': batch_min,
        'maxs': batch_max
    }

@tf.function
def calc_lin_interp_coeffs(t, inner_t):
    """
    Computes interpolation coefficients k_0 and k_1 for given time points;
    used then for linear interpolation by lin_nd_sampler()

    Parameters:
    - t: External time points (1D tensor)
    - inner_t: Inner time points for which interpolation is required (1D tensor)

    Returns:
    - k_0: Coefficient for the left time points during interpolation
    - k_1: Coefficient for the right time points during interpolation
    - idx_0: Index of the time points on the left of inner time points
    - idx_1: Index of the time points on the right of inner time points
    """

    # Ensure the rank and size of t and inner_t is fine
    tf.debugging.assert_rank(t, 1, message="t is not a 1D tensor")
    tf.debugging.assert_greater(
        tf.shape(t)[0], 2, message="Size of t is not greater than 2"
    )
    tf.debugging.assert_rank(inner_t, 1, message="inner_t is not a 1D tensor")
    # tf.debugging.assert_greater(
    #     tf.shape(inner_t)[0], 1, message="Size of inner_t is not greater than 1"
    # )

    # Compute the indices for interpolation
    idx_1 = tf.searchsorted(t, inner_t, side='right')
    idx_0 = idx_1 - 1
    t1 = tf.gather(t, idx_1)
    t0 = tf.gather(t, idx_0)

    # Compute the interpolation coefficients
    k_0 = (inner_t - t0) / (t1 - t0)
    k_1 = 1 - k_0

    # Ensure k_0 and k_1 are between [0, 1]
    tf.debugging.assert_greater_equal(k_0, 0.0, message="k_0 has values less than 0")
    tf.debugging.assert_less_equal(k_0, 1.0, message="k_0 has values greater than 1")
    tf.debugging.assert_greater_equal(k_1, 0.0, message="k_1 has values less than 0")
    tf.debugging.assert_less_equal(k_1, 1.0, message="k_1 has values greater than 1")

    return k_0, k_1, idx_0, idx_1

@tf.function
def lin_nd_sampler(data, k_0, k_1, idx_0, idx_1, frame_length):
    """Interpolates multidimensional data 1D interpolation coefficients found by calc_lin_interp_coeffs().
    Parameters
    ----------
    data : tf.Tensor
        A 3D tensor representing the data.
    k_0, k_1, idx_0, idx_1: tf.Tensor
        linear interpolation coefficients
    frame_length: int
        the number of frames (necessary for the tf.function to work well)
    Returns
    -------
    output : tf.Tensor
        A tensor with the same shape as `data`, but with values interpolated at the specified time points.
    """

    shape = data.shape
    data_0 = tf.gather(data, idx_0, axis=1)

    if data.dtype in [tf.int32, tf.int64]:
        output = data_0
    else:
        data_1 = tf.gather(data, idx_1, axis=1)
        k_0 = tf.identity(tf.expand_dims(k_0, 0))
        k_1 = tf.identity(tf.expand_dims(k_1, 0))
        k_0 = tf.expand_dims(k_0, -1)
        k_1 = tf.expand_dims(k_1, -1)

        if tf.rank(data) == 4:
            k_0 = tf.expand_dims(k_0, -1)
            k_1 = tf.expand_dims(k_1, -1)

        k_0 = tf.broadcast_to(k_0, tf.shape(data_0))
        k_1 = tf.broadcast_to(k_1, tf.shape(data_0))
        output = k_0 * data_0 + k_1 * data_1

    output.set_shape([shape[0]] + [frame_length] + shape[2:])
    return output

@tf.function
def define_trajectory(
    sample,
    frame_length=3,
    num_interpolator_nodes=10,
    left_pad=True,
    num_equally_spaced_frames=None,
    raw_graph=False,
    ):
    # initialization
    trajectory = {key: tf.identity(sample['trajectory'][key])
                  for key in sample['trajectory']}

    trajectory_dict = {}
    for k_1 in sample['trajectory_dict']:
        trajectory_dict[k_1] = {}
        for k_2 in sample['trajectory_dict'][k_1]:
            trajectory_dict[k_1][k_2] = tf.identity(sample['trajectory_dict'][k_1][k_2])

    trajectory_length = tf.identity(sample['trajectory_length'])

    ########## THIS PART CAN BE EDITTED FOR FUTURE STUDIES ##########

    # defining the node types and keeping only the data for 1 frame.
    node_type = trajectory['lf_node_type']
    node_type_shape = node_type.shape
    node_type = tf.cast(node_type, tf.float32)[:, 0:1, :]
    node_type_shape = [node_type_shape[0], 1, node_type_shape[-1]]
    node_type.set_shape(node_type_shape)

    # definining the trajectory
    outputs, outputs_dict = {}, {}
    outputs['node_features'] = {
        # 's': tf.concat(
        #     [trajectory['hf_depth']],
        #     axis=-1,
        #     ),
        'v': tf.concat(
            [trajectory['hf_vec_1'], trajectory['hf_vec_2']],
            axis=-1,
            ),
        't': tf.concat(
            [trajectory['lf_stress'], trajectory['lf_strain']],
            axis=-1,
            ),
        }
    outputs_dict['edges'] = {'s': trajectory_dict['lf_edges']}

    if not raw_graph:
        # outputs['nearest_nodes'] = {'s': trajectory['lf_nearest_nodes'][:,:,:num_interpolator_nodes]}
        outputs_dict['augmented_senders'] = {'s': trajectory_dict['lf_augmented_senders']}
        outputs_dict['augmented_receivers'] = {'s': trajectory_dict['lf_augmented_receivers']}
        outputs_dict['augmented_dx'] = {'v': trajectory_dict['lf_augmented_dx']}
        outputs_dict['augmented_x0'] = {'v': trajectory_dict['lf_augmented_x0']}
    else:
        outputs['pos'] = {'v': trajectory['lf_pos']}

    time = tf.squeeze(trajectory['hf_time'])

    outputs['target'] = {
        's': tf.concat([
              trajectory['hf_pore_pressure'],
              trajectory['hf_gag_stress'],
              trajectory['hf_fibrilar_stress_mises'],
              trajectory['hf_non_fibrilar_stress_mises'],
              ], axis=-1,
              ),
        # 't': tf.concat(
        #     [trajectory['hf_stress'], trajectory['hf_strain']],
        #     axis=-1,
        #     ),
        # 's': tf.concat([
        #         trajectory['hf_stress_mises'],
        #         ], axis=-1,
        #      ),
        }
    # outputs['target'] = {
    #     't': trajectory['lf_stress'],
    #     # 's': tf.cast(trajectory['lf_node_type'], dtype=tf.float32)
    #     }
    # outputs['node_features'] = {'t': tf.identity(trajectory['lf_stress'])}

    ########################################################################

    # repeating the initial frames
    if left_pad == True:
        num = trajectory_length - frame_length

        if num > 0:
            num = 1
        else:
            num = -1 * num

        for k2 in outputs_dict:
            for k3 in outputs_dict[k2]:
                for k4 in outputs_dict[k2][k3]:
                    data = outputs_dict[k2][k3][k4]
                    tf.assert_equal(tf.shape(data)[1], trajectory_length)
                    new_frames = tf.repeat(data[:, 0:1], repeats=num, axis=1)
                    data = tf.concat([new_frames, data], axis=1)
                    outputs_dict[k2][k3][k4] = data

        for k2 in outputs:
            for k3 in outputs[k2]:
                data = outputs[k2][k3]
                tf.assert_equal(tf.shape(data)[1], trajectory_length)
                new_frames = tf.repeat(data[:, 0:1], repeats=num, axis=1)
                data = tf.concat([new_frames, data], axis=1)
                outputs[k2][k3] = data

        delta = (time[-1:] - time[0:1])/(tf.cast(trajectory_length, tf.float32))
        delta *= tf.range(num, 0, -1, tf.float32)
        new_initial_time = time[0:1] - delta
        time = tf.concat([new_initial_time, time], axis=0)-new_initial_time[0]
        trajectory_length += num

    if num_equally_spaced_frames != None:
        t = tf.linspace(time[0], time[-1], num_equally_spaced_frames)
        inner_t = t[1:-1]
        k_0, k_1, idx_0, idx_1 = calc_lin_interp_coeffs(time, inner_t)
        # k_0, k_1 = k_0[tf.newaxis, :, tf.newaxis], k_1[tf.newaxis, :, tf.newaxis]
        time = tf.concat([time[0:1], inner_t, time[-1:]], axis=0)

        for k1 in outputs:
            for k2 in outputs[k1]:
                data = outputs[k1][k2]
                inner_data = lin_nd_sampler(data, k_0, k_1, idx_0, idx_1, num_equally_spaced_frames-2)
                outputs[k1][k2] = tf.concat(
                    [data[:,:1], inner_data, data[:,-1:]], axis=1
                    )

        for k1 in outputs_dict:
            for k2 in outputs_dict[k1]:
                for k3 in outputs_dict[k1][k2]:
                    data = outputs_dict[k1][k2][k3]
                    inner_data = lin_nd_sampler(data, k_0, k_1, idx_0, idx_1, num_equally_spaced_frames-2)
                    outputs_dict[k1][k2][k3] = tf.concat(
                        [data[:,:1], inner_data, data[:,-1:]], axis=1
                        )

        trajectory_length = num_equally_spaced_frames

    return {
        'trajectory': outputs,
        'trajectory_dict': outputs_dict,
        'trajectory_length': trajectory_length,
        'node_type': node_type,
        'time': time,
        }

@tf.function
def parse(proto, meta):
    meta = copy.deepcopy(meta)
    tensor_list = {
        k: tf.io.VarLenFeature(tf.string) for k in meta['array_names']
    }
    tensors = tf.io.parse_single_example(proto, tensor_list)
    trajectory_length = tensors.pop('trajectory_length')

    meta_trajectory_length = meta['arrays'].pop('trajectory_length')
    trajectory_length = tf.io.decode_raw(
        trajectory_length.values, getattr(tf, meta_trajectory_length['dtype'])
    )
    trajectory_length = tf.cast(tf.squeeze(trajectory_length), tf.int32)

    out_1 = {}
    for key, field in meta['arrays'].items():
        data = tf.io.decode_raw(
            tensors[key].values, getattr(tf, field['dtype'])
        )
        data = tf.reshape(data, field['shape'])
        if data.shape[1] == 1:
            data = tf.tile(data, [1, trajectory_length, 1])
        out_1[key] = data

    out_2 = {}
    for key_1 in sorted(meta['dicts']):
        out_2[key_1] = {}
        for key_2 in meta['dicts'][key_1]:
            key = key_1 + '_' + str(key_2)
            field = out_1[key]
            del out_1[key]
            out_2[key_1][key_2] = field

    return {
        'trajectory': out_1,
        'trajectory_dict': out_2,
        'trajectory_length': trajectory_length
    }

@tf.function
def rotate_3d_symmetric_tensor(tensor, r_all):
    """Performs a rotation operation on a 3D symmetric tensor.

    This function assumes the last dimension of the tensor represents a flattened symmetric matrix of
    shape [6], and reshapes it to a 2D symmetric matrix of shape [3, 3]. The function then applies
    a rotation matrix 'r_all' to this 2D matrix.

    Parameters
    ----------
    tensor : tf.Tensor
        The tensor to be rotated. The last dimension must be divisible by 6.
    r_all : tf.Tensor
        The rotation matrix to be applied.

    Returns
    -------
    rotated_tensor : tf.Tensor
        The rotated tensor, with the same shape as the input tensor.
    """
    shape = tensor.shape
    dim = 6
    tensor = tf.reshape(tensor, shape[:-1] + [shape[-1] // dim, dim])

    # Reshape the tensor to a [3, 3] matrix
    tensor = tf.stack([
        tf.stack([tensor[..., 0], tensor[..., 3], tensor[..., 4]], axis=-1),
        tf.stack([tensor[..., 3], tensor[..., 1], tensor[..., 5]], axis=-1),
        tf.stack([tensor[..., 4], tensor[..., 5], tensor[..., 2]], axis=-1),
    ], axis=-1)

    # Apply the rotation
    tensor = tf.matmul(tf.transpose(r_all), tf.matmul(tensor, r_all))

    # Reshape the tensor back to a flat [6] vector
    tensor = tf.stack(
        [tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2],
         tensor[..., 0, 1], tensor[..., 0, 2], tensor[..., 1, 2]],
        axis=-1
    )

    return tf.reshape(tensor, shape)

@tf.function
def rotate_3d_vector(vector, r_all):
    """Rotates a 3D vector using a rotation matrix.

    This function reshapes the last dimension of the vector to [3] (assuming it is divisible by 3),
    then applies a rotation matrix to it.

    Parameters
    ----------
    vector : tf.Tensor
        The 3D vector to be rotated. The last dimension must be divisible by 3.
    r_all : tf.Tensor
        The rotation matrix to be applied.

    Returns
    -------
    rotated_vector : tf.Tensor
        The rotated vector, with the same shape as the input vector.
    """
    shape = vector.shape
    dim = 3
    vector = tf.reshape(vector, shape[:-1] + [shape[-1] // dim, dim])

    # Apply the rotation
    vector = tf.matmul(vector, r_all)

    return tf.reshape(vector, shape)

@tf.function
def generate_random_3d_rotation():
    """Generates a random 3D rotation matrix.

    This function generates a random 3D rotation matrix using the ZYX intrinsic rotation
    convention (i.e., first a rotation about the z-axis, then the y-axis, and finally the x-axis).

    Returns
    -------
    rotation_matrix : tf.Tensor
        The generated 3D rotation matrix.
    """
    two_pi = 2 * math.pi

    # Generate random Euler angles
    roll = tf.random.uniform([], 0, two_pi)  # Rotation about x-axis
    pitch = tf.random.uniform([], 0, two_pi)  # Rotation about y-axis
    yaw = tf.random.uniform([], 0, two_pi)  # Rotation about z-axis

    # Compute trigonometric functions
    cos_roll = tf.cos(roll)
    cos_pitch = tf.cos(pitch)
    cos_yaw = tf.cos(yaw)
    sin_roll = tf.sin(roll)
    sin_pitch = tf.sin(pitch)
    sin_yaw = tf.sin(yaw)

    # Construct the rotation matrix
    rotation_matrix = tf.stack([
        [
            cos_yaw * cos_pitch,
            cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
            cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
        ],
        [
            sin_yaw * cos_pitch,
            sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
            sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
        ],
        [
            -sin_pitch,
            cos_pitch * sin_roll,
            cos_pitch * cos_roll
        ],
    ])

    return rotation_matrix

@tf.function
def generate_frames(
    sample,
    frame_length=3,
    rotation=True,
    interpolation=True,
    concat_previous_targets=True,
    add_time=True,
    component='all'
    ):
    """Generates frames from graph data.

    This function applies several preprocessing steps to a sample of graph data, including
    temporal interpolation, data rotation, and feature concatenation.

    Parameters
    ----------
    sample : dict
        The sample of graph data.

    frame_length : int, optional
        The number of frames to generate, by default 3.

    rotation : bool, optional
        If true, applies a random rotation to the graph data, by default True.

    interpolation : bool, optional
        If true, applies temporal interpolation to the graph data, by default True.

    concat_previous_targets : bool, optional
        If true, concatenates previous targets into the current sample's feature set, by default True.

    add_time : bool, optional
        If true, adds a temporal feature to the node features, by default True.

    component : str, optional
        The component to select from the target variable, by default 'all'.

    Returns
    -------
    dict
        The processed version of the input data sample.
    """
    def _do_nothing(data):
        return data

    def _keep_last(data):
        shape = data.shape
        data = data[:, -1:]
        data.set_shape([shape[0]] + [1] + shape[2:])
        return data

    keys = list(sample['trajectory']) + list(sample['trajectory_dict'])

    if concat_previous_targets == True:
        frame_length += 1
        sample['trajectory']['target_pre'] = {}

        for k, val in sample['trajectory']['target'].items():
            sample['trajectory']['target_pre'][k] = tf.identity(val)

        def _shift_forward(data):
            shape = data.shape
            data = data[:, 1:]
            data.set_shape([shape[0]] + [frame_length-1] + shape[2:])
            return data

        def _shift_backward(data):
            shape = data.shape
            data = data[:, :-1]
            data.set_shape([shape[0]] + [frame_length-1] + shape[2:])
            return data

        _fix_frame = {key: _shift_forward for key in keys}
        _fix_frame['target'] = _keep_last
        _fix_frame['target_pre'] = _shift_backward
    else:
        _fix_frame = {key: _do_nothing for key in keys}

    if interpolation == True:
        time = sample['time']
        eps = 0.0000001
        dt = tf.random.uniform([], minval=time[-1]/1000, maxval=time[-1]/10)
        t = tf.sort(
            tf.random.uniform([], minval=eps, maxval=(time[-1]-dt-eps)) +
            tf.random.uniform([frame_length], minval=0.0, maxval=dt)
            )
        k_0, k_1, idx_0, idx_1 = calc_lin_interp_coeffs(time, t)
        # k_0, k_1 = k_0[tf.newaxis, :, tf.newaxis], k_1[tf.newaxis, :, tf.newaxis]

        def _temporal_sampler(data):
            return lin_nd_sampler(data, k_0, k_1, idx_0, idx_1, frame_length)

    else:
        num_frame = sample['trajectory_length']
        maxval=num_frame-frame_length-1
        idx = tf.random.uniform(
            shape=(), minval=0, maxval=maxval, dtype=tf.int32
            )
        t = sample['time'][idx:idx+frame_length]

        def _temporal_sampler(data):
            data = data[:, idx:idx+frame_length]
            shape = data.shape
            shape = [shape[0]] + [frame_length] + shape[2:]
            data.set_shape(shape)
            return data

    # rotational transformation, if applicable

    if rotation == True:
        r_all = generate_random_3d_rotation()

        def _rotate_3d_vector(data):
            return rotate_3d_vector(data, r_all=r_all)

        def _rotate_3d_symmetric_tensor(data):
            return rotate_3d_symmetric_tensor(data, r_all=r_all)

        _rotate = {
            'v': _rotate_3d_vector,
            't': _rotate_3d_symmetric_tensor,
            }
    else:
        _rotate = {
            'v': _do_nothing,
            't': _do_nothing,
            }

    _rotate['s'] = _do_nothing

    trajectory, trajectory_dict = {}, {}

    for k1 in sample['trajectory']:
        trajectory[k1] = []

        for k2 in sample['trajectory'][k1]:
            data = _temporal_sampler(sample['trajectory'][k1][k2])
            data = _fix_frame[k1](data)
            data = _rotate[k2](data)
            trajectory[k1].append(data)

        trajectory[k1] = tf.concat(trajectory[k1], axis=-1)

    # similar to the above loop but first get the innermost
    # keys (k3) and initialize for each seperately.
    for k1 in sample['trajectory_dict']:
        trajectory_dict[k1] = {}

        for k2 in sample['trajectory_dict'][k1]:
            for k3 in sample['trajectory_dict'][k1][k2]:
                trajectory_dict[k1][k3] = []

        for k3 in trajectory_dict[k1]:
            for k2 in sample['trajectory_dict'][k1]:
                data = sample['trajectory_dict'][k1][k2][k3]
                data = _temporal_sampler(sample['trajectory_dict'][k1][k2][k3])
                data = _fix_frame[k1](data)
                data = _rotate[k2](data)
                trajectory_dict[k1][k3].append(data)

            trajectory_dict[k1][k3] = tf.concat(
                trajectory_dict[k1][k3], axis=-1
                )

    if add_time == True:
        relative_t = tf.subtract(t, t[-1])

        if concat_previous_targets == True:
            relative_t = relative_t[:-1]

        relative_t = tf.tile(
            relative_t[tf.newaxis,:,tf.newaxis],
            (trajectory['node_features'].shape[0], 1, 1)
            )
        shape = trajectory['node_features'].shape
        trajectory['node_features'] = tf.concat(
            [trajectory['node_features'], relative_t], axis=2
            )
        trajectory['node_features'].set_shape(shape[:-1]+[shape[-1]+1])

    if component != 'all':
        lst = ['target']
        if concat_previous_targets == True:
            lst += ['target_pre']

        for key in lst:
            trajectory[key] = trajectory[key][:, :, component:component+1]

    if concat_previous_targets == True:
        target_pre = trajectory.pop('target_pre')
        trajectory['node_features'] = tf.concat(
            [trajectory['node_features'], target_pre], axis=-1
            )
    else:
        trajectory['target'] = _keep_last(trajectory['target'])

    node_type = tf.tile(sample['node_type'], [1, frame_length, 1])

    return {
        'trajectory': trajectory,
        'trajectory_dict': trajectory_dict,
        'node_type': node_type,
        }

@tf.function
def graph_sum(sample, batch_size):
    """
    This function performs batching by reshaping and reindexing the graph tensors.

    Args:
        sample (dict): A dictionary containing information about the graph such as
                       trajectory, trajectory_dict, edges, augmented_nodes, augmented_dx, and augmented_x0.
        batch_size (int): The size of the batch which will be used for reindexing tensors in the graph.

    Returns:
        sample (dict): The input dictionary after reshaping and reindexing tensors based on batch size.
    """
    summed_sample = {'trajectory_dict': {}, 'trajectory': {}}
    edges = sample['trajectory_dict']['edges']
    trajectory = sample['trajectory']
    augmented_senders = sample['trajectory_dict']['augmented_senders']
    augmented_receivers = sample['trajectory_dict']['augmented_receivers']
    augmented_dx = sample['trajectory_dict']['augmented_dx']
    augmented_x0 = sample['trajectory_dict']['augmented_x0']

    # e.g., if we have 1280 nodes >> 1279
    max_node_idx = tf.math.reduce_max(
        tf.concat(list(edges.values()), axis=1),
        )
    # e.g., >> [0, 1, 2]
    multiplier = tf.range(batch_size)
    # e.g., >> [0, 1280, 2560]
    added_idx = max_node_idx * multiplier + multiplier
    # for broadcasting
    added_idx = added_idx[:, tf.newaxis, tf.newaxis, tf.newaxis]

    def update_shape(t):
        reshaped_t = tf.reshape(t, tf.concat([[-1], t.shape[2:]], axis=0))
        reshaped_t.set_shape(reshaped_t.shape)
        return reshaped_t


    # first adding the edges indeces, fixing and updating its shape

    edges = {key: tf.identity(edges[key]) for key in edges}
    for key in edges:
        edges[key] += added_idx
        edges[key] = update_shape(edges[key])

    summed_sample['trajectory_dict']['edges'] = edges

    # then adding the indeces of the augmented nodes, reshaping them

    augmented_senders = {
        key: tf.identity(augmented_senders[key]) for key in augmented_senders
        }
    augmented_receivers = {
        key: tf.identity(augmented_receivers[key]) for key in augmented_receivers
        }
    reshaped_augmented_senders, reshaped_augmented_receivers = {}, {}
    reshaped_augmented_dx, reshaped_augmented_x0 = {}, {}
    for key in augmented_senders:
        augmented_senders[key] += added_idx
        reshaped_augmented_senders[key] = update_shape(augmented_senders[key])
        augmented_receivers[key] += added_idx
        reshaped_augmented_receivers[key] = update_shape(augmented_receivers[key])
        reshaped_augmented_dx[key] = update_shape(augmented_dx[key])
        reshaped_augmented_x0[key] = update_shape(augmented_x0[key])

    summed_sample['trajectory_dict']['augmented_x0'] = reshaped_augmented_x0
    summed_sample['trajectory_dict']['augmented_senders'] = reshaped_augmented_senders
    summed_sample['trajectory_dict']['augmented_receivers'] = reshaped_augmented_receivers
    summed_sample['trajectory_dict']['augmented_dx'] = reshaped_augmented_dx

    # fixing the shapes of others
    for key in ['node_features', 'target']:
        summed_sample['trajectory'][key] = update_shape(trajectory[key])

    node_type = tf.identity(sample['node_type'])
    summed_sample['node_type'] = update_shape(node_type)

    # # updating indeces and shape of nearest nodes.
    # nearest_nodes = update_shape(trajectory['nearest_nodes'] + added_idx)
    # summed_sample['trajectory']['nearest_nodes'] = nearest_nodes

    return summed_sample

@tf.function
def define_edges(nodal_position, senders, receivers):
    """
    Defines the edge features based on node positions, senders and receivers.

    The edge features are represented as the difference in the positions of the sender
    and receiver nodes (relative position), and the Euclidean norm of this relative position.

    Args:
        nodal_position (tf.Tensor): Tensor representing the positions of nodes in a graph.
        senders (tf.Tensor): Tensor representing the indices of sender nodes for each edge.
        receivers (tf.Tensor): Tensor representing the indices of receiver nodes for each edge.

    Returns:
        edge_features (tf.Tensor): Tensor representing the features of each edge,
                                   including relative position and its Euclidean norm.
    """
    relative_pos = (
        tf.gather(nodal_position, senders) -
        tf.gather(nodal_position, receivers)
        )
    edge_norm = tf.norm(relative_pos, axis=-1, keepdims=True)
    edge_features = tf.concat([relative_pos, edge_norm], axis=-1)

    return edge_features

@tf.function
def normalize_graph(normalizable_data, means, stds, epsilon=1e-7):
    """
    Normalize data by subtracting the mean and dividing by the standard deviation.

    Args:
        normalizable_data (dict): A dictionary of data arrays to be normalized.
                                  Each key is a data label, and each value is a tensor of data.
        means (dict): A dictionary of mean values for each data array.
        stds (dict): A dictionary of standard deviation values for each data array.
        epsilon (float): A small constant added to the standard deviation to prevent division by zero.

    Returns:
        dict: A dictionary of normalized data.
    """
    normalized_data = {}

    for key in normalizable_data:
        mean_tensor = tf.reshape(means[key], [1, 1, -1]) # for broadcasting
        std_tensor = tf.reshape(stds[key], [1, 1, -1]) # for broadcasting

        normalized_data[key] = tf.math.divide(
            tf.math.subtract(normalizable_data[key], mean_tensor),
            std_tensor + epsilon
        )

    return normalized_data

@tf.function
def make_graph(
    sample,
    maxK=1,
    transformation=None,
    return_nodal_pos=True,
    # noises_std=0.1,
    flatten=True,
    ):
    """
    Constructs a graph representation of the input data based on the given trajectories and their properties.

    For nodes with more than one connection, this function adjusts their position by randomly selecting a connection
    and displacing the node along this edge. The resulting graph will have updated nodal positions and potentially
    augmented features.

    Parameters:
    ----------
    sample : dict
        A dictionary containing trajectories and their associated attributes. The dictionary structure includes:
        - trajectory: Contains node_features, nearest_nodes, and target.
        - trajectory_dict: Contains edges, augmented_nodes, augmented_dx, and augmented_x0.

    maxK : int, optional
        Maximum value for the random multiplier used in updated position calculation. Default is 1.

    transformation : None or dict, optional
        If None, return without transformation. If a dictionary, it should have the stats for normalization.

    return_nodal_pos : bool, optional
        If True, the resulting dictionary will include the nodal positions in the features.

    noises_std : float, optional
        Standard deviation of the noise to be added to the node features. Default is 0.1.

    flatten : bool, optional
        If True, the last two dimensions of the features and outputs will be flattened.

    Returns:
    -------
    dict
        A dictionary representing the constructed graph. It includes:
        - node features
        - edge features
        - sender and receiver indices
        - targets
        - (potentially) nodal positions

    -------
    Note: Nodes with a key of 0 in the 'augmented_nodes' dict are not augmented.
    """
    trajectory = sample['trajectory']
    # finding the updated positions after augmentation
    interpolated_features = {'node_features': [], 'target': []}
    senders_pos, x0, senders = [], [], []
    trajectory_dict = sample['trajectory_dict']
    for k1 in trajectory_dict['augmented_senders']:
        augmented_x0 = trajectory_dict['augmented_x0'][k1][:, :, 0]
        augmented_dx = trajectory_dict['augmented_dx'][k1]
        senders.append(trajectory_dict['augmented_senders'][k1][:, 0, 0])
        x0.append(augmented_x0)
        sender_features = {
            k: tf.gather(trajectory[k], senders[-1], axis=0) for k in interpolated_features
            }
        if k1 == 0:
            # no aumentation
            senders_pos.append(augmented_x0)
            for k2 in interpolated_features:
                interpolated_features[k2].append(sender_features[k2])
        else:
            # defining augmentation coefficient k and new sender positions
            shape = augmented_dx.shape[:1]
            idx = tf.random.uniform(shape=shape, minval=0, maxval=int(k1), dtype=tf.int32)
            dx = tf.gather(augmented_dx, idx, axis=2, batch_dims=1)
            k = tf.random.uniform(shape=shape, minval=0, maxval=maxK, dtype=tf.float32)
            k = k[..., tf.newaxis, tf.newaxis]
            senders_pos.append(augmented_x0 + k * dx)
            # feature interpolation
            augmented_receivers = tf.gather(
                trajectory_dict['augmented_receivers'][k1],
                idx,
                axis=2,
                batch_dims=1
                )[:, 0]
            receiver_features = {
                k2: tf.gather(trajectory[k2], augmented_receivers, axis=0)
                for k2 in interpolated_features
                }
            for k2 in interpolated_features:
                interpolated_features[k2].append(
                    (1-k)*sender_features[k2] + k*receiver_features[k2]
                    )

    # Fixing the shape and indeces of augmented nodal positions
    senders_pos = tf.concat(senders_pos, axis=0)
    senders = tf.concat(senders, axis=0)
    senders_sort_args = tf.argsort(senders)
    updated_pos = tf.gather(senders_pos, senders_sort_args, axis=0)

    # Similarly fixing the shape and indeces of interpolated features
    for k in interpolated_features:
         interpolated_features[k] = tf.concat(interpolated_features[k], axis=0)
         interpolated_features[k] = tf.gather(
            interpolated_features[k], senders_sort_args, axis=0
            )
    node_features = interpolated_features['node_features']
    # outputs = {'targets': interpolated_features['target']}
    target = interpolated_features['target']

    # finding previous nodal positions
    x0 = tf.concat(x0, axis=0)
    x0 = tf.gather(x0, senders_sort_args, axis=0)

    # finding senders and recievers
    edges = sample['trajectory_dict']['edges']
    keys = sorted(edges.keys())
    senders = [edges[key][:, 0, 0] for key in keys]
    receivers = [edges[key][:, 0, 1] for key in keys]
    senders = tf.concat(senders, axis=0)
    receivers = tf.concat(receivers, axis=0)
    senders_sort_args = tf.argsort(senders)
    senders = tf.gather(senders, senders_sort_args)
    receivers = tf.gather(receivers, senders_sort_args)

    # # finding the nearest points and their w.
    # nearest_nodes = sample['trajectory']['nearest_nodes'][:, 0]
    # nearest_nodes_pos = tf.gather(x0, nearest_nodes, axis=0)
    # nearest_nodes_norm = tf.math.reduce_euclidean_norm(
    #     nearest_nodes_pos - updated_pos[:, tf.newaxis], axis=-1
    #     )
    # w = tf.math.reciprocal(nearest_nodes_norm + 1e-6)
    # normalized_w = w/tf.math.reduce_sum(w, axis=-1, keepdims=True)

    # # finding updated node features
    # nearest_node_features = tf.gather(
    #     sample['trajectory']['node_features'], nearest_nodes, axis=0
    #     )
    # node_features = tf.reduce_sum(normalized_w[..., tf.newaxis] * nearest_node_features, axis=1)
    #
    # # similarly for the target features
    # nearest_target_features = tf.gather(
    #     sample['trajectory']['target'], nearest_nodes, axis=0
    #     )
    # targets = tf.reduce_sum(normalized_w[..., tf.newaxis] * nearest_target_features, axis=1)
    # outputs = {'targets': targets}

    # adding gradual noises
    # if noises_std > 0.0:
    #     node_features += tf.random.normal(node_features.shape, stddev=noises_std)

    # adding node_types into features with no augmentation
    node_features = tf.concat([node_features, sample['node_type']], axis=-1)

    # applying normalization and making the (sub-)graph
    features = {'node_features': node_features, 'targets': target}
    edges = {'edge_features': define_edges(updated_pos, senders, receivers)}

    if transformation is not None:
        features = transform(features, **transformation)
        edges = transform(edges, keys = ['edge_features'], **transformation)

    features['edge_features'] = edges['edge_features']

    if return_nodal_pos == True:
        features['nodal_pos'] = updated_pos

    if flatten == True:
        for key in features:
            last_dim = tf.constant([features[key].shape[-1] * features[key].shape[-2]])
            new_shape = tf.concat([features[key].shape[:-2], last_dim], axis=-1)
            features[key] = tf.reshape(features[key], new_shape)

    outputs = {'targets': features.pop('targets')}
    # concatenating features and topology as inputs
    inputs = features
    inputs['senders'] = senders
    inputs['receivers'] = receivers

    return {**inputs, **outputs}

@tf.function
def fix_keras_dimension(sample):
    """
    Expand the dimensions of input tensors for compatibility with Keras.

    This function is designed to fix some of the issues with tensor dimensions
    that may arise when using Keras. It expands the dimensions of the tensors
    in the input dictionary, adding an extra first dimension, as Keras
    expects the first dimension of the input tensors to represent the batch size.

    Args:
        sample (dict): A dictionary of tensors that represent the input.
        autoencoder (bool, optional): If True, returns the specified tensor as both input and output. Defaults to False.

    Returns:
        tuple: The processed input tensor and the corresponding target tensor,
        or the specified tensor as both input and output if 'autoencoder' is True.
    """

    out = {}
    if 'target_nodes' not in sample:
        out['target_nodes'] = tf.range(sample['targets'].shape[0])

    out.update(sample)

    # adding a redundant axes to avoid issues in keras during testing.
    for k in out:
        out[k] = tf.expand_dims(out[k], axis=0)

        if k in ['senders', 'receivers', 'target_nodes']:
            out[k] = tf.expand_dims(out[k], axis=-1)
            out[k].set_shape([1, None, 1])
        else:
            shape = out[k].shape
            out[k].set_shape([1, None, shape[-1]])

    out_targets = out.pop('targets')
    return out, out_targets

    # if autoencoder == False:
    #     out_targets = out.pop('targets')
    #     return out, out_targets
    # else:
    #     return out[autoencoder], out[autoencoder]

@tf.function
def get_target_probability(tensor, min_val=-2, max_val=+2, num_bins=50):
    """
    Calculate the probability distribution of a tensor based on its binning.

    This function creates a histogram with predefined edges using TensorFlow Probability's
    find_bins function, counts the occurrences of unique values (bins), and calculates
    their probabilities.

    Args:
        tensor (tf.Tensor): The tensor for which the probability distribution is calculated.
        min_val (float): The lower edge of the first bin in the histogram.
        max_val (float): The upper edge of the last bin in the histogram.
        num_bins (int): The number of bins in the histogram.

    Returns:
        tf.Tensor: The calculated probability distribution of the tensor.
    """
    bin_edges = tf.cast(tf.linspace(min_val, max_val, num_bins), dtype=tf.float32)
    bins = tfp.stats.find_bins(tensor, bin_edges, dtype=tf.int32)
    bins = tf.squeeze(bins)
    uniques, bins, counts = tf.unique_with_counts(bins)
    probabilities = 1/counts
    probabilities = probabilities/tf.reduce_sum(probabilities)
    return tf.gather(probabilities, bins, axis=0)

@tf.function
def subgrapher(
    sample,
    probabilities,
    num_target_nodes,
    sub_nodes,
    sub_receivers,
    sub_senders,
    sub_edges,
    sub_local_receivers,
    sub_local_senders,
    local_node_target,
    num_relevant_nodes,
    num_relevant_edges,
    ):
    """
    Extract a subgraph based on a target node chosen according to the given probabilities.

    Args:
        sample (dict): A dictionary containing keys 'targets', 'node_features', 'edge_features'
            representing the entire graph.
        probabilities (tf.Tensor): A tensor of probabilities for each node to be chosen
            as the target node.
        num_target_nodes (int): The number of target nodes to choose.
        subgraph_nodes, subgraph_receivers, subgraph_senders, subgraph_edges (tf.Tensor):
            Tensors containing information about the nodes, receivers, senders, and edges
            of the subgraphs.
        subgraph_local_receivers, subgraph_local_senders (tf.Tensor): Tensors containing
            information about the local receivers and senders of the subgraphs.
        local_node_target (tf.Tensor): A tensor mapping each node to its corresponding target.
        num_relevant_nodes, num_relevant_edges (tf.Tensor): Tensors specifying the number
            of nodes and edges relevant to each target node.

    Returns:
        dict: A dictionary containing the extracted subgraph with keys 'targets',
            'node_features', 'edge_features', 'target_nodes', 'receivers', and 'senders'.
    """
    targets = sample['targets']
    node_features = sample['node_features']
    edge_features = sample['edge_features']

    # getting the target nodes
    log_probabilities = tf.math.log(probabilities)
    target_node = tf.random.categorical(
        tf.reshape(log_probabilities, [1, -1]),
        num_target_nodes,
        dtype=tf.int32
        )
    target_node = tf.squeeze(target_node)

    # getting the node, edge, and target features
    out = {}
    out['targets'] = tf.gather(targets, target_node, axis=0)
    relevant_nodes = tf.reshape(tf.gather(sub_nodes, target_node, axis=0), [-1])

    out['node_features'] = tf.gather(node_features, relevant_nodes, axis=0)
    relevant_edges = tf.reshape(tf.gather(sub_edges, target_node, axis=0), [-1])
    out['edge_features'] = tf.gather(edge_features, relevant_edges, axis=0)

    # getting and fixing topolgy
    added_idx = tf.gather(num_relevant_nodes, target_node, axis=0)
    added_idx = tf.math.cumsum(added_idx)[:-1]
    added_idx = tf.concat([[0], added_idx], axis=0)

    out['target_nodes'] = added_idx + tf.gather(
        local_node_target, target_node, axis=0
        )
    added_idx = tf.expand_dims(added_idx, axis=-1)
    out['receivers'] = tf.reshape(
        added_idx + tf.gather(sub_local_receivers, target_node, axis=0), [-1]
        )
    # tf.print(out['receivers'].shape)
    out['senders'] = tf.reshape(
        added_idx + tf.gather(sub_local_senders, target_node, axis=0), [-1]
        )
    return out

@tf.function
def get_weight(data, num_bins=50):
    """
    Calculate the weight of each value in the data tensor. The weight is inversely proportional
    to the frequency of the value.

    Args:
        data (tf.Tensor): The input data tensor.
        num_bins (int): The number of bins to be used for histogram.

    Returns:
        tf.Tensor: The tensor of weights of the same shape as the input data tensor.
    """

    def _get_bins(data):
        """
        Categorize the values in the data tensor into bins.

        Args:
            data (tf.Tensor): The input data tensor.

        Returns:
            tf.Tensor: The tensor of bin indices of the same shape as the input data tensor.
        """
        edges = tf.linspace(tf.reduce_min(data), tf.reduce_max(data), num_bins)
        return tfp.stats.find_bins(data, edges, dtype=tf.int32)

    bins = tf.map_fn(
        _get_bins, tf.transpose(data), dtype=tf.int32, parallel_iterations=6
        )
    _, bins = tf.raw_ops.UniqueV2(x=tf.transpose(bins), axis=[0])
    _, _, count = tf.unique_with_counts(bins)
    count = tf.repeat(count, repeats=count)
    # returning to original order
    count = tf.gather(count, tf.argsort(tf.argsort(bins)))
    frequency = 1.0 / tf.cast(count, tf.float32)
    return frequency / tf.reduce_sum(frequency)

def load_and_split_ds(
    modelName='Simple_3D',
    test_split=0.33,
    valid_split=0.33,
    leftover_split=0,
    dataset_dir='gnn_datasets',
    ):
    """
    Load a dataset from a TFRecord file and split it into training, validation, and testing sets.

    Args:
        modelName (str): The name of the model associated with the dataset.
        test_split (float): The fraction of the total number of samples to use for testing.
        valid_split (float): The fraction of the total number of samples to use for validation.
        leftover_split (float): The fraction of the total number of samples to reserve as "leftovers".
        dataset_dir (str): The directory where the dataset and its metadata are stored.

    Returns:
        dict: A dictionary with keys 'train', 'valid', and 'test' and values being the respective tf.data.Dataset instances.
        dict: A dictionary with keys 'train', 'valid', and 'test' and values being the number of samples in each set.
        str: The path to the metadata file.
    """
    path_fn = partial(os.path.join, dataset_dir, modelName)
    meta_path = path_fn('meta.json')
    with open(meta_path, 'r') as fp:
        meta = json.loads(fp.read())

    ds = tf.data.TFRecordDataset(path_fn('data.tfrecord'))
    num_samples = meta['total_num_samples']
    num_leftover = int(num_samples * leftover_split)
    num = {
        'test': int(num_samples * test_split),
        'valid': int(num_samples * valid_split),
        }
    num['train'] = num_samples - num['test'] - num['valid'] - num_leftover
    ds = ds.shuffle(buffer_size=num_samples, seed=123)
    ds = {
        'test': ds.take(num['test']),
        'valid': ds.skip(num['test']).take(num['valid']),
        'train': ds.skip(num['test'] + num['valid']).take(num['train']),
        }

    return ds, num, meta_path

def preprocessor_init(
    left_pad,
    generate_frames_params,
    num_equally_spaced_frames,
    graph_batch_size,
    make_graph_params,
    model_dir='knee',
    ):
    """
    Initialize a preprocessor for a data processing pipeline.

    Args:
        left_pad (bool): An argument for define_trajectory function.
        generate_frames_params (dict): A set of parameters for generate_frames function.
        num_equally_spaced_frames (int): Number of equally spaced frames for define_trajectory function.
        graph_batch_size (dict): The batch size for each set in the dataset.
        make_graph_params (dict): A set of parameters for make_graph function.

    Returns:
        dict: A dictionary with keys 'train', 'valid', and 'test' and values being the respective preprocessed tf.data.Dataset instances.
        tf.data.Dataset: A single graph dataset derived from the training set.
        dict: A dictionary of functions to further preprocess each set in the dataset.
    """
    generate_frames_params = generate_frames_params.copy()

    ds, num, meta_path = load_and_split_ds(modelName=model_dir)

    with open(meta_path, 'r') as fp:
        meta = json.loads(fp.read())

    define_trajectory_func = lambda proto: define_trajectory(
        parse(proto, meta=meta),
        left_pad=left_pad,
        frame_length=generate_frames_params['frame_length'],
        num_equally_spaced_frames=num_equally_spaced_frames
        )
    batch_size = {}

    for k in ds:

        # if (num[k] * repeat_num[k] // graph_batch_size[k]) < 1:
        #     batch_size[k] = num[k] * repeat_num[k]
        # else:
        #     batch_size[k] = graph_batch_size[k]

        batch_size[k] = graph_batch_size[k]

        ds[k] = ds[k].map(define_trajectory_func).cache()

        # if k == 'test':
        #     original_ds_test = ds['test']

        ds[k] = ds[k].shuffle(num[k]).repeat()

    generate_training_frames_params = generate_frames_params.copy()
    generate_frames_func = {
        'train': partial(generate_frames, **generate_training_frames_params)
        }
    generate_frames_params['rotation'] = False
    generate_frames_params['interpolation'] = False
    func = partial(generate_frames, **generate_frames_params)
    generate_frames_func['valid'] = generate_frames_func['test'] = func

    preprocess_func = {}

    func = lambda sample: make_graph(
        generate_frames_func['train'](sample), **make_graph_params
        )
    ds_one_graph = ds['train'].map(func, num_parallel_calls=auto)

    for k in ds:
        ds[k] = ds[k].map(generate_frames_func[k], num_parallel_calls=auto)
        ds[k] = ds[k].batch(
            batch_size[k], num_parallel_calls=auto, drop_remainder=True
            )
        preprocess_func[k] = partial(graph_sum, batch_size=batch_size[k])

    return ds, ds_one_graph, preprocess_func

def preprocessor_final(
    autoencoder,
    make_graph_params,
    ds_normalized,
    ds_one_graph,
    preprocess_func,
    graph_batch_size,
    # frame_length,
    num_message_passing_layers=1,
    # resampling=True,
    num_target_nodes=10
    ):
    """
    This function prepares the data for the model.

    Parameters:
    autoencoder (bool): Flage for autoencoder models.
    make_graph_params (dict): parameters used to generate graph representation of the data.
    ds (dict): a dictionary containing train, test, and validation datasets.
    ds_one_graph (dict): a dataset containing a single graph.
    preprocess_func (function): function to preprocess the data.
    graph_batch_size (int): size of the graph batch.
    frame_length (int): length of the frame for the data.
    num_message_passing_layers (int, optional): number of message passing layers in the model. Default is 1.
    resampling (bool, optional): whether to resample the data or not. Default is True.
    num_target_nodes (int, optional): number of target nodes in the graph. Default is 10.

    Returns:
    ds (dict): a dictionary containing prepared train, test, and validation datasets.
    subgrapher_func (function): a function to extract subgraphs from a larger graph. Only returned if resampling is True.
    targets (np.array): array of target values. Only returned if resampling is True.
    """

    def make_graph_fn(sample):
        return make_graph(preprocess_func['train'](sample), **make_graph_params)


    ds = {}

    ds['train_sample'] = ds_normalized['train'].map(make_graph_fn, num_parallel_calls=auto,)


    for key in ['train', 'valid', 'test']:
        def fix_keras_dimension_fn(sample):
            return fix_keras_dimension(
                make_graph(preprocess_func[key](sample), **make_graph_params),
                )
        ds[key] = ds_normalized[key].map(fix_keras_dimension_fn, num_parallel_calls=auto)

    # in case a gpu is available
    for key in ['train_sample', 'train', 'valid']:
        ds[key] = ds[key].prefetch(auto)

    # getting the resampling data with a sample used in normalization

    for sample in ds_one_graph.take(1):
        senders = sample.pop('senders')
        receivers = sample.pop('receivers')

    node_features = sample['node_features']
    edge_features = sample['edge_features']
    targets = sample['targets']

    # getting the subgraphs on each node
    num_nodes = node_features.shape[0]
    num_edges = edge_features.shape[0]

    nodes = tf.range(num_nodes) # node labels
    edges = tf.range(num_edges) # edge labels

    senders = senders.numpy()
    receivers = receivers.numpy()
    node_features = node_features.numpy()
    edge_features = edge_features.numpy()

    # Define some helper functions
    def first_aggregation(node):
        # getting the edges' labels conneced to the node
        s = np.where(senders == node)[0]
        r = np.where(receivers == node)[0]
        return np.union1d(r, s)

    def get_neighbors(node):
        # (from topology) getting the neighboring nodes for the node
        indeces = np.where(senders == node)[0]
        return receivers[indeces]

    def next_aggregation(neighbors, edges):
        # aggregate edges' label from the neighboring nodes of the node
        new_aggregated_edges = np.concatenate(edges[neighbors], axis=-1)
        return np.unique(new_aggregated_edges)

    edges = [first_aggregation(node) for node in nodes]
    neighbors = [get_neighbors(node) for node in nodes]

    for _ in range(num_message_passing_layers + 1):
        edges = np.array(edges, dtype=object)
        edges = [next_aggregation(neighbors[node], edges) for node in nodes]

    sub_senders = []
    sub_receivers = []
    sub_nodes = []
    sub_local_senders = []
    sub_local_receivers = []
    local_node_target = []
    num_relevant_edges = []
    num_relevant_nodes = []

    for node in nodes:
        relevant_edges = np.sort(edges[node])
        relevant_senders = senders[relevant_edges]
        relevant_receivers = receivers[relevant_edges]
        relevant_nodes = np.union1d(relevant_senders, relevant_receivers)
        relevant_nodes = np.sort(relevant_nodes)

        edges[node] = relevant_edges
        sub_senders.append(relevant_senders)
        sub_receivers.append(relevant_receivers)
        sub_nodes.append(relevant_nodes)
        sub_local_senders.append(
            np.searchsorted(np.unique(relevant_senders), relevant_senders)
            )
        sub_local_receivers.append(
            np.searchsorted(np.unique(relevant_receivers), relevant_receivers)
            )
        local_node_target.append(
            np.squeeze(np.where(node == relevant_nodes))
            )
        num_relevant_edges.append(relevant_edges.shape[0])
        num_relevant_nodes.append(relevant_nodes.shape[0])

    get_ragged_tensor = lambda val, length: tf.RaggedTensor.from_row_lengths(
        values=np.concatenate(val, dtype=np.int32), row_lengths=length
        )

    sub_nodes = get_ragged_tensor(sub_nodes, num_relevant_nodes)
    sub_receivers = get_ragged_tensor(sub_receivers, num_relevant_edges)
    sub_senders = get_ragged_tensor(sub_senders, num_relevant_edges)
    sub_edges = get_ragged_tensor(edges, num_relevant_edges)
    sub_local_senders = get_ragged_tensor(sub_local_senders, num_relevant_edges)
    sub_local_receivers = get_ragged_tensor(sub_local_receivers, num_relevant_edges)
    local_node_target = tf.constant(local_node_target, dtype=tf.int32)
    num_relevant_nodes = tf.constant(num_relevant_nodes, dtype=tf.int32)

    added_node_labels = tf.repeat(
        tf.range(0, num_nodes*graph_batch_size['train'], num_nodes),
        graph_batch_size['train']*[num_nodes]
        )
    added_node_labels = added_node_labels[:, tf.newaxis]
    sub_nodes = tf.concat(graph_batch_size['train']*[sub_nodes], axis=0)
    sub_nodes += added_node_labels
    sub_receivers = tf.concat(graph_batch_size['train']*[sub_receivers], axis=0)
    sub_receivers += added_node_labels
    sub_senders = tf.concat(graph_batch_size['train']*[sub_senders], axis=0)
    sub_senders += added_node_labels
    sub_local_senders = tf.concat(
        graph_batch_size['train']*[sub_local_senders], axis=0
        )
    sub_local_receivers = tf.concat(
        graph_batch_size['train']*[sub_local_receivers], axis=0
        )
    num_relevant_nodes = tf.concat(
        graph_batch_size['train']*[num_relevant_nodes], axis=0
        )
    local_node_target = tf.concat(
        graph_batch_size['train']*[local_node_target], axis=0
        )
    added_edge_labels = tf.repeat(
        tf.range(0, num_edges*graph_batch_size['train'], num_edges),
        graph_batch_size['train']*[num_nodes]
        )
    sub_edges = tf.concat(graph_batch_size['train']*[sub_edges], axis=0)
    sub_edges += added_edge_labels[:, tf.newaxis]

    subgrapher_func = partial(
        subgrapher,
        num_target_nodes=num_target_nodes,
        sub_nodes=sub_nodes,
        sub_receivers=sub_receivers,
        sub_senders=sub_senders,
        sub_edges=sub_edges,
        sub_local_receivers=sub_local_receivers,
        sub_local_senders=sub_local_senders,
        local_node_target=local_node_target,
        num_relevant_nodes=num_relevant_nodes,
        num_relevant_edges=num_relevant_edges,
        )

    return ds, subgrapher_func

def tensor_to_numpy(obj):
    if isinstance(obj, dict):
        return {k: tensor_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, tf.Tensor):
        return obj.numpy()
    else:
        return obj

def numpy_to_tensor(obj):
    if isinstance(obj, dict):
        return {k: numpy_to_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return tf.convert_to_tensor(obj)
    else:
        return obj

def load_transform_stats(name):
    with open(f"{name}.pkl", "rb") as file:
        loaded_data = pickle.load(file)

    return numpy_to_tensor(loaded_data)

def save_transform_stats(name, data):
    with open(f"{name}.pkl", "wb") as file:
        pickle.dump(data, file)

class top_k_loss(tf.keras.losses.Loss):
    def __init__(self, k):
        super().__init__()
        self.k = k
    def call(self, y_true, y_pred):
        return top_k_error(y_true, y_pred, self.k)

class weighted_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.weight = None
    def call(self, y_true, y_pred):
        error = tf.math.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis = -1)
        return error * self.weight

class ScalarMeanSquaredError(tf.keras.losses.Loss):

    def __init__(self, name="scalar_mean_squared_error", **kwargs):
        super(ScalarMeanSquaredError, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        mse_losses = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return tf.reduce_mean(mse_losses)


class LinearIncreaseScheduler(
    tf.keras.optimizers.schedules.LearningRateSchedule
    ):
    def __init__(self, initial_lr, final_lr, total_steps):
        super(LinearIncreaseScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps

    def __call__(self, step):
        lr_diff = self.final_lr - self.initial_lr
        return self.initial_lr + tf.minimum(step / self.total_steps, 1.0) * lr_diff


def train_step(x, y, model, loss_func, optimizer, x_valid, y_valid):
    with tf.GradientTape() as tape:
        predict_train = model(x)[0]
        loss = loss_func(y[0], predict_train)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    predict_valid = model(x_valid)[0]
    return loss, predict_train, predict_valid

@tf.function
def top_k_error(y_true, y_pred, k):
    error = tf.math.abs(y_pred - y_true)
    error = tf.reshape(error, [-1])
    error, _ = tf.math.top_k(error, k=k, sorted=False)
    return tf.reduce_mean(error)

def errors_with_training(
    x, y, predict_train, x_valid, y_valid, k_valid, predict_valid,
    ):
    def _smse(x, y):
        """Scaler MSE metric"""
        return tf.reduce_mean(tf.keras.metrics.mean_squared_error(x, y))

    max_error_valid = top_k_error(y_valid[0], predict_valid, k_valid)
    mse_train = _smse(y[0], predict_train)
    mse_valid = _smse(y_valid[0], predict_valid)

    return mse_train, mse_valid, max_error_valid


class model_tools:
    """Our custmom loop containing the ML models and the custom loop,
    plus few utilities to (re)store data."""
    def __init__(
        self,
        transformation = None,
        frame_length = 3,
        rotation = True,
        interpolation = True,
        concat_previous_targets = False,
        component = 'all',
        maxK = 1.0,
        flatten = True,
        return_nodal_pos = False,
        left_pad = False,
        num_equally_spaced_frames = 10,
        folder_name = 'res',
        reset_all = False,
        model_dir='knee',
        graph_batch_size = {'test': 5, 'valid': 1, 'train': 1}
        ):

        self.model = None

        if reset_all == True:
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)

            os.makedirs(folder_name)

        generate_frames_params = {
            'rotation': rotation,
            'interpolation': interpolation,
            'concat_previous_targets': concat_previous_targets,
            'component': 'all',
            'frame_length': frame_length,
            }
        make_graph_params = {
            'return_nodal_pos': return_nodal_pos,
            'maxK': maxK,
            # 'noises_std': 0.0,
            'flatten': flatten,
            'transformation': transformation
            }
        ds_normalized, ds_one_graph, preprocess_func = preprocessor_init(
            left_pad,
            generate_frames_params,
            num_equally_spaced_frames,
            graph_batch_size,
            make_graph_params,
            model_dir=model_dir,
            )
        self.frame_length = frame_length
        self.maxK = maxK
        self.graph_batch_size = graph_batch_size
        self.make_graph_params = make_graph_params
        self.ds_normalized = ds_normalized
        self.ds_one_graph = ds_one_graph
        self.preprocess_func = preprocess_func
        self.folder_name = folder_name
        self.encoder = None

    def prepare_ds(
        self,
        autoencoder='node_features',
        num_message_passings=0,
        num_target_nodes=1000,
        ):
        ds, subgrapher_func = preprocessor_final(
            autoencoder,
            self.make_graph_params,
            self.ds_normalized,
            self.ds_one_graph,
            self.preprocess_func,
            self.graph_batch_size,
            num_message_passing_layers=num_message_passings,
            num_target_nodes=num_target_nodes
            )

        self.num_message_passings = num_message_passings
        self.subgrapher_func = subgrapher_func
        self.ds = ds
        self.autoencoder = autoencoder

    def initialize_training(
        self,
        compression_ratio=4,
        exp_num = 1,
        loss_type = 'top_10',
        lr = 0.1,
        main_patience = 2,
        patience = 100,
        noise_std = 0.01,
        num_layers = 1,
        model_type = 'gnn',
        # latent_size = 8,
        weight_sharing=False,
        ):
        ds = self.ds
        
        for x_valid, y_valid in ds["valid"].take(1): pass
        for x_test, y_test in ds["test"].take(1): pass

        latent_size = int(x_valid['node_features'].shape[-1]) // compression_ratio

        if model_type == 'dae':
            self.make_dae(
                bottleneck_dim=latent_size, num_deep_layers=num_layers
                )
            y_test = x_test['node_features']
            y_valid = x_valid['node_features']
        else:
            self.make_gnn(
                latent_size=latent_size,
                num_mlp_layers=num_layers,
                weight_sharing=weight_sharing,
                model_type=model_type
                )

        if model_type == 'combined':
            if self.encoder is None:
                raise ValueError("self.encoder is None!")
            x_test['node_features'] = self.encoder(x_test)[tf.newaxis]
            x_valid['node_features'] = self.encoder(x_valid)[tf.newaxis]

        if loss_type[-6:] == 'weight':
            loss_func = weighted_loss()
        elif loss_type[:3] == 'top':
            num_nodes = ds["train"].element_spec[-1].shape[1]
            if loss_type == 'top_10':
                k = num_nodes//10
            elif loss_type == 'top_1':
                k = num_nodes//100
            loss_func = top_k_loss(k)
        else:
            loss_func = ScalarMeanSquaredError()

        self.recorded_data = {
            'model_type': model_type,
            'frame_length': self.frame_length,
            'exp_num': exp_num,
            'weight_sharing': weight_sharing,
            'num_message_passings': self.num_message_passings,
            'latent_size': latent_size,
            'loss_type': loss_type,
            'patience': patience,
            'main_patience': main_patience,
            'compression_ratio': compression_ratio,
            'mse_train' : [],
            'mse_valid' : [],
            'max_error_valid' : [],
            'lr': lr,
            'num_layers': num_layers,
            'patience': patience,
            'graph_batch_size': self.graph_batch_size['train'],
            'maxK': self.maxK,
            'noise_std': noise_std,
            }
        self.x_valid, self.y_valid = x_valid, y_valid
        self.x_test, self.y_test = x_test, y_test
        self.model_type = model_type
        self.noise_std = noise_std
        self.lr = lr
        self.loss_func = loss_func
        self.loss_type = loss_type
        self.patience = patience
        self.main_patience = main_patience

    def make_gnn(
        self,
        latent_size=8,
        num_mlp_layers=1,
        weight_sharing=False,
        residual=True,
        model_type='gnn'
        ):
        """Message passing graph neural network"""

        if self.num_message_passings == 0:
            raise ValueError("self.num_message_passings is 0 for gnn!")

        num_message_passings = self.num_message_passings

        inputs_dict = {}
        for x_train, y_train in self.ds['train'].take(1):
            for key in x_train:
                if key == 'node_features' and model_type == 'combined':
                    x_train[key] = self.encoder(x_train)
                inputs_dict[key] = Input(
                    shape=([None] + [x_train[key].shape[-1]]),
                    dtype=x_train[key].dtype,
                    name=key
                    )
        output_size = y_train.shape[-1]

        class MLP(Layer):
            """layer of multi-layer perceptrons with selu activation function"""
            def __init__(self, num_mlp_dense_layers=num_mlp_layers, output_size=None):
                super(MLP, self).__init__()
                layers = []
                for _ in range(num_mlp_dense_layers):
                    layer = Dense(
                        units=latent_size,
                        activation='selu',
                        kernel_initializer='lecun_normal'
                        )
                    layers.append(layer)
                    # layers.append(LayerNormalization())

                if output_size != None:
                    # for decoder
                    layer = Dense(
                        units=output_size,
                        activation=None,
                        kernel_initializer='lecun_normal'
                        )
                    layers.append(layer)

                self.layers = layers

            def call(self, inputs):
                for layer in self.layers:
                    inputs = layer(inputs)

                return inputs
        

        # features, shape: [None, None, ...]
        nodes = inputs_dict['node_features'][0]
        edges_mesh =  inputs_dict['edge_features'][0]

        # connections, shape: [None]
        senders_mesh = inputs_dict['senders'][0, :, 0]
        receivers_mesh = inputs_dict['receivers'][0, :, 0]

        # number of nodes
        num_nodes = tf.shape(nodes)[0]

        # message passing layers

        def aggregate_nodes(nodes, senders, receivers):
            return [
                tf.gather(nodes, senders, axis=0),
                tf.gather(nodes, receivers, axis=0)
                ]

        def aggregate_edges(edges, receivers):
            return [tf.math.unsorted_segment_mean(edges, receivers, num_nodes)]

        def update_features(current_features, aggregated_features, fn_approximator):
            return fn_approximator(
                tf.concat([current_features] + aggregated_features, axis=-1)
                )

        nodes = MLP()(nodes)
        edges_mesh = MLP()(edges_mesh)

        for step in range(1, num_message_passings+1):

            if weight_sharing == True:
                if step == 1:
                    mlp_nodes = MLP()
                    mlp_edges = MLP()
            else:
                mlp_nodes = MLP()
                mlp_edges = MLP()

            # aggregation:
            aggregated_node_features = aggregate_edges(
                edges_mesh, receivers_mesh
                )
            aggregated_mesh_edge_features = aggregate_nodes(
                nodes, senders_mesh, receivers_mesh
                )

            # updating:
            nodes_updated = update_features(
                nodes, aggregated_node_features, mlp_nodes
                )
            edges_mesh_updated = update_features(
                edges_mesh, aggregated_mesh_edge_features, mlp_edges
                )

            # adding residual connections:
            if residual == True:
                edges_mesh += edges_mesh_updated
                nodes += nodes_updated
            else:
                edges_mesh = edges_mesh_updated
                nodes = nodes_updated

        nodes = MLP(output_size=output_size)(nodes)

        # target nodes
        target_nodes = inputs_dict['target_nodes'][0, :, 0]

        # boolean tensor
        mask = tf.tensor_scatter_nd_update(
            tf.zeros_like(nodes[:, 0], dtype=tf.bool),
            tf.reshape(target_nodes, [-1, 1]),
            tf.ones_like(target_nodes, dtype=tf.bool)
            )
        nodes = tf.boolean_mask(nodes, mask)

        output = nodes[tf.newaxis]

        self.model = Model(inputs=inputs_dict, outputs=output)

    def make_dae(self, bottleneck_dim=8, num_deep_layers=1):
        """denoising autoencoder"""
        dense_args = {'activation': 'selu', 'kernel_initializer': 'lecun_normal'}
        element_spec_dict = self.ds['train'].element_spec[0]
        bottleneck_dim = bottleneck_dim

        item = element_spec_dict['node_features']
        num_dims = item.shape[-1]

        # encoder
        encoder_input = {}
        for key, item in element_spec_dict.items():
            encoder_input[key] = Input(
                shape=len(item.shape[1:-1])*[None] + [item.shape[-1]],
                dtype=item.dtype,
                name=key
                )

        x = encoder_input['node_features'][0]

        for _ in range(num_deep_layers):
            x = Dense(num_dims, **dense_args)(x)

        for _ in range(num_deep_layers):
            x = Dense(bottleneck_dim, **dense_args)(x)

        encoder_output = x

        encoder = Model(encoder_input, encoder_output)

        # decoder
        decoder_input = Input(shape=[bottleneck_dim])
        x = decoder_input

        for _ in range(num_deep_layers):
            x = Dense(num_dims, **dense_args)(x)

        decoder_output = Dense(
            units=num_dims, activation=None, kernel_initializer='lecun_normal'
            )(x)
        decoder = Model(decoder_input, decoder_output)

        # autoencoder
        autoencoder_input = encoder_input
        target_nodes = autoencoder_input['target_nodes'][0, :, 0]

        # boolean tensor
        decoded = decoder(encoder(autoencoder_input))
        mask = tf.tensor_scatter_nd_update(
            tf.zeros_like(
                autoencoder_input['node_features'][0, :, 0], dtype=tf.bool
                ),
            tf.reshape(target_nodes, [-1, 1]),
            tf.ones_like(target_nodes, dtype=tf.bool)
            )
        masked_decoded = tf.boolean_mask(decoded, mask)
        decoder_output_expanded = masked_decoded[tf.newaxis, ...]
        autoencoder = Model(autoencoder_input, decoder_output_expanded)

        self.encoder = encoder
        self.decoder = decoder
        self.model = autoencoder
        
        self.bottleneck_dim = bottleneck_dim

    def _smse(self, x, y):
        """Scaler MSE metric"""
        return tf.reduce_mean(tf.keras.metrics.mean_squared_error(x, y))

    def _train_step_fn(self, x, y, x_valid, y_valid, k_valid):

        model, loss_func, optimizer = self.model, self.loss_func, self.optimizer
        with tf.GradientTape() as tape:
            predict_train = model(x)[0]
            loss = loss_func(y[0], predict_train)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        predict_valid = model(x_valid)[0]

        max_error_valid = top_k_error(y_valid[0], predict_valid, k_valid)
        mse_train = self._smse(y[0], predict_train)
        mse_valid = self._smse(y_valid[0], predict_valid)

        return loss, mse_train, mse_valid, max_error_valid

    def fit(self, max_epochs=100):
        ds = self.ds
        lr = self.lr
        noise_std = self.noise_std
        patience = self.patience
        loss_type = self.loss_type
        x_valid, y_valid = self.x_valid, self.y_valid
        x_test, y_test = self.x_test, self.y_test
        model = self.model
        loss_func = self.loss_func
        main_patience = self.main_patience
        _train_step_fn = self._train_step_fn
        subgrapher_func = self.subgrapher_func
        model_type = self.model_type
        best_model_params = model.get_weights()
        k_valid = tf.reshape(y_valid, [-1]).shape[-1]//100
        lr_schedule = LinearIncreaseScheduler(
            initial_lr=lr/10.0, final_lr=lr, total_steps=patience
            )

        epoch = 0
        main_early_stopping = 0

        start_time = time.time()

        for sample in ds['train_sample'].take(max_epochs):
            epoch += 1
            step = 0
            early_stopping = 0

            if model_type == 'dae':
                sample['targets'] = tf.identity(sample['node_features'])
            
            if model_type in ['dae', 'gnn']:
                noise = tf.random.normal(sample['node_features'].shape, mean=0, stddev=noise_std)
                sample['node_features'] += noise
            
            x, y = fix_keras_dimension(sample)

            if model_type == 'combined':
                x['node_features'] = self.encoder(x)[tf.newaxis]

            if loss_type == 'static_weight':
                loss_func.weight = get_weight(y[0], num_bins=2)
                self.loss_func = loss_func
            elif loss_type == 'static_subgraph':
                probabilities = get_target_probability(
                    tf.reduce_mean(sample['targets'], axis=-1, keepdims=True)
                    )

            self.optimizer = tf.keras.optimizers.Adam()

            if epoch == 1:
                current_lr = lr_schedule(step)
                self.optimizer.lr.assign(current_lr)

                if loss_type[:7] == 'dynamic':
                    error = tf.math.abs(y[0] - model(x)[0])
                    error = tf.reduce_mean(error, axis=-1, keepdims=True)
                    weights = get_weight(error)
                    if loss_type == 'dynamic_weight':
                        loss_func.weight = weights
                        self.loss_func = loss_func
                    else:
                        probabilities = weights
                if loss_type in ['static_subgraph', 'dynamic_subgraph']:
                    x, y = fix_keras_dimension(
                        subgrapher_func(sample=sample, probabilities=probabilities),
                        )
                    if model_type == 'dae':
                        y = tf.gather(x['node_features'], x['target_nodes'][0, :, 0], axis=1)
                    elif model_type == 'combined':
                        x['node_features'] = self.encoder(x)[tf.newaxis]

                loss, mse_train, mse_valid, max_error_valid = _train_step_fn(
                    x, y, x_valid, y_valid, k_valid
                    )
                best_max_mse_valid = max_error_valid
                best_mse_valid = mse_valid
                best_mse_train = mse_train
                self.recorded_data['mse_train'].append(mse_train)
                self.recorded_data['mse_valid'].append(mse_valid)
                self.recorded_data['max_error_valid'].append(best_max_mse_valid)
                continue
            else:
                if loss_type[:7] == 'dynamic':
                    error = tf.math.abs(y[0] - model(x)[0])
                    error = tf.reduce_mean(error, axis=-1, keepdims=True)
                    weights = get_weight(error)
                    if loss_type == 'dynamic_weight':
                        loss_func.weight = weights
                        self.loss_func = loss_func
                    else:
                        probabilities = weights

                if loss_type in ['static_subgraph', 'dynamic_subgraph']:
                    x, y = fix_keras_dimension(
                        subgrapher_func(sample=sample, probabilities=probabilities),
                        )
                    if self.model_type == 'dae':
                        y = tf.gather(x['node_features'], x['target_nodes'][0, :, 0], axis=1)
                    elif model_type == 'combined':
                        x['node_features'] = self.encoder(x)[tf.newaxis]
            
            while early_stopping < patience:
                current_lr = lr_schedule(step)
                self.optimizer.lr.assign(current_lr)

                loss, mse_train, mse_valid, max_error_valid = _train_step_fn(
                    x, y, x_valid, y_valid, k_valid
                    )
                if max_error_valid < (0.99 * best_max_mse_valid):
                    best_model_params = model.get_weights()
                    best_max_mse_valid = max_error_valid
                    best_mse_valid = mse_valid
                    best_mse_train = mse_train
                    early_stopping = 0
                    main_early_stopping = 0
                elif max_error_valid > (2 * best_max_mse_valid):
                    model.set_weights(best_model_params)
                    max_error_valid = best_max_mse_valid
                    mse_valid = best_mse_valid
                    mse_train = best_mse_train
                    break
                else:
                    early_stopping += 1

                step += 1

                tf.print(f"Step: {step}")
                tf.print(f"Epoch {epoch}/{max_epochs}")
                tf.print("Learning rate: {:.7f}".format(current_lr))
                tf.print(f"Loss: {loss}")
                tf.print("Error (valid): {:.4f}".format(mse_valid.numpy()))
                tf.print("Valid max error: {:.4f}".format(max_error_valid.numpy()))
                tf.print("Best max error: {:.4f}".format(best_max_mse_valid.numpy()))
                tf.print(f"Early stopping: {early_stopping}")
                tf.print('\n')

            if main_early_stopping > main_patience:
                lr_schedule.initial_lr = lr_schedule.initial_lr / 10
                lr_schedule.final_lr = lr_schedule.final_lr / 10
                main_early_stopping = 0
            else:
                main_early_stopping += 1

            self.recorded_data['mse_train'].append(mse_train)
            self.recorded_data['mse_valid'].append(mse_valid)
            self.recorded_data['max_error_valid'].append(best_max_mse_valid)
            
            if lr_schedule.initial_lr < 1e-5:
                break
        
        end_time = time.time()
        self.recorded_data['time'] = end_time - start_time
        model.set_weights(best_model_params)
        self.model = model

    def store_data(self):

        recorded_data = self.recorded_data
        file_name = f"params_{recorded_data['loss_type']}_{recorded_data['exp_num']}_{recorded_data['model_type']}"
        data_address = os.path.join(self.folder_name, file_name + ".pkl")

        with open(data_address, "wb") as f:
            for k in ['mse_valid', 'mse_train', 'max_error_valid']:
                data = tf.stack(recorded_data[k], axis=0)
                recorded_data[k] = data.numpy().tolist()

            pickle.dump(recorded_data, f)

        if self.model_type == 'combined':
            weights_address = os.path.join(self.folder_name, file_name + "_rom.h5")
            self.encoder.save_weights(weights_address)
            weights_address = os.path.join(self.folder_name, file_name + "_gnn.h5")
            self.model.save_weights(weights_address)
        else:
            weights_address = os.path.join(self.folder_name, file_name + ".h5")
            self.model.save_weights(weights_address)

    def load_data(self, loss_type, exp_num, model_type):

        file_name = f"params_{loss_type}_{exp_num}_{model_type}"
        address = os.path.join(self.folder_name, file_name)

        with open(address + ".pkl", "rb") as file:
            recorded_data = pickle.load(file)

        latent_size = recorded_data['latent_size']
        num_layers = recorded_data['num_layers']

        if model_type == 'gnn':
            self.make_gnn(latent_size=latent_size, num_mlp_layers=num_layers)
            self.model.load_weights(address + ".h5")
        else:
            self.make_dae(bottleneck_dim=latent_size, num_deep_layers=num_layers)
            if model_type == 'combined':
                self.encoder.load_weights(address + "_rom.h5")
                self.make_gnn(latent_size=latent_size, num_mlp_layers=num_layers, model_type='combined')
                self.model.load_weights(address + "_gnn.h5")
            else:
                self.model.load_weights(address + ".h5")

        self.recorded_data = recorded_data
        return recorded_data