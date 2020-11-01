import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

            # vertex_text
            vertex_F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            vertex_geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            vertex_angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid,
                                            normalizer_fn=None) - 0.5) * np.pi / 2
            vertex_F_geometry = tf.concat([vertex_geo_map, vertex_angle_map], axis=-1)

            #vertex_text
            vertex_1_F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            vertex_1_geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            vertex_1_angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2
            vertex_1_F_geometry = tf.concat([vertex_1_geo_map, vertex_1_angle_map], axis=-1)

            #vertex_text_
            vertex_2_F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            vertex_2_geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            vertex_2_angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2
            vertex_2_F_geometry = tf.concat([vertex_2_geo_map, vertex_2_angle_map], axis=-1)

            #vertex_text_
            vertex_3_F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            vertex_3_geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            vertex_3_angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2
            vertex_3_F_geometry = tf.concat([vertex_3_geo_map, vertex_3_angle_map], axis=-1)

            #vertex_text_
            vertex_4_F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            vertex_4_geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            vertex_4_angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2
            vertex_4_F_geometry = tf.concat([vertex_4_geo_map, vertex_4_angle_map], axis=-1)



    return F_score, F_geometry, vertex_F_score, vertex_F_geometry, vertex_1_F_score, vertex_1_F_geometry, vertex_2_F_score, vertex_2_F_geometry, vertex_3_F_score, vertex_3_F_geometry, vertex_4_F_score, vertex_4_F_geometry


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss

def focal_loss(y_true_cls, y_pred_cls):
    '''
    :param y_true_cls:
    :param y_pred_cls:
    :return:
    '''
    gamma = 2
    alpha = 0.5

    dim = tf.reduce_prod(tf.shape(y_true_cls)[1:])
    flat_y_true_cls = tf.reshape(y_true_cls, [-1, dim])
    flat_y_pred_cls = tf.reshape(y_pred_cls, [-1, dim])
    pt = flat_y_true_cls*flat_y_pred_cls+(1.0-flat_y_true_cls)*(1.0 - flat_y_pred_cls)
    weight_map = alpha*tf.pow((1.0-pt),gamma)
    weighted_loss = tf.multiply(((flat_y_true_cls * tf.log(flat_y_pred_cls + 1e-9)) + ((1 - flat_y_true_cls) * tf.log(1 - flat_y_pred_cls + 1e-9))), weight_map)
    cross_entropy = -tf.reduce_sum(weighted_loss,axis = 1)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('classification_focal_loss', cross_entropy_mean)
    return cross_entropy_mean



def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask,
         vertex_y_true_cls, vertex_y_pred_cls, vertex_y_true_geo,vertex_y_pred_geo,vertex_training_mask,
         vertex_1_y_true_cls, vertex_1_y_pred_cls, vertex_1_y_true_geo, vertex_1_y_pred_geo, vertex_1_training_mask,
         vertex_2_y_true_cls, vertex_2_y_pred_cls, vertex_2_y_true_geo, vertex_2_y_pred_geo, vertex_2_training_mask,
         vertex_3_y_true_cls, vertex_3_y_pred_cls, vertex_3_y_true_geo, vertex_3_y_pred_geo, vertex_3_training_mask,
         vertex_4_y_true_cls, vertex_4_y_pred_cls, vertex_4_y_true_geo, vertex_4_y_pred_geo, vertex_4_training_mask):
# def loss(y_true_cls, y_pred_cls,
#          y_true_geo, y_pred_geo,
#          training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = focal_loss(y_true_cls, y_pred_cls)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    vertex_classification_loss = focal_loss(vertex_y_true_cls, vertex_y_pred_cls)
    vertex_classification_loss *= 0.01

    vertex_1_classification_loss = focal_loss(vertex_1_y_true_cls, vertex_1_y_pred_cls)
    vertex_1_classification_loss *= 0.01
    vertex_2_classification_loss = focal_loss(vertex_2_y_true_cls, vertex_2_y_pred_cls)
    vertex_2_classification_loss *= 0.01
    vertex_3_classification_loss = focal_loss(vertex_3_y_true_cls, vertex_3_y_pred_cls)
    vertex_3_classification_loss *= 0.01
    vertex_4_classification_loss = focal_loss(vertex_4_y_true_cls, vertex_4_y_pred_cls)
    vertex_4_classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    w_pre = abs(d1_pred)+abs(d3_pred)
    h_pre = abs(d2_pred)+abs(d4_pred)
    # L_AABB = L_AABB_IOU + abs_smooth(abs(abs(d1_gt)-abs(d1_pred))/w_pre+abs(abs(d2_gt)-abs(d2_pred))/h_pre+abs(abs(d3_gt)-abs(d3_pred))/w_pre+abs(abs(d4_gt)-abs(d4_pred)))/h_pre
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    # vertex_d1 -> top, d2->right, d3->bottom, d4->left
    vertex_d1_gt, vertex_d2_gt, vertex_d3_gt, vertex_d4_gt, vertex_theta_gt = tf.split(value=vertex_y_true_geo, num_or_size_splits=5, axis=3)
    vertex_d1_pred, vertex_d2_pred, vertex_d3_pred, vertex_d4_pred, vertex_theta_pred = tf.split(value=vertex_y_pred_geo, num_or_size_splits=5, axis=3)
    vertex_area_gt = (vertex_d1_gt + vertex_d3_gt) * (vertex_d2_gt + vertex_d4_gt)
    vertex_area_pred = (vertex_d1_pred + vertex_d3_pred) * (vertex_d2_pred + vertex_d4_pred)
    vertex_w_union = tf.minimum(vertex_d2_gt, vertex_d2_pred) + tf.minimum(vertex_d4_gt, vertex_d4_pred)
    vertex_h_union = tf.minimum(vertex_d1_gt, vertex_d1_pred) + tf.minimum(vertex_d3_gt, vertex_d3_pred)
    vertex_area_intersect = vertex_w_union * vertex_h_union
    vertex_area_union = vertex_area_gt + vertex_area_pred - vertex_area_intersect
    vertex_L_AABB = -tf.log((vertex_area_intersect + 1.0)/(vertex_area_union + 1.0))
    vertex_L_theta = 1 - tf.cos(vertex_theta_pred - vertex_theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(vertex_L_AABB * vertex_y_true_cls * vertex_training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(vertex_L_theta * vertex_y_true_cls * vertex_training_mask))
    vertex_L_g = vertex_L_AABB + 20 * vertex_L_theta


    # vertex_d1 -> top, d2->right, d3->bottom, d4->left
    vertex_1_d1_gt, vertex_1_d2_gt, vertex_1_d3_gt, vertex_1_d4_gt, vertex_1_theta_gt = tf.split(value=vertex_1_y_true_geo,
                                                                                       num_or_size_splits=5, axis=3)
    vertex_1_d1_pred, vertex_1_d2_pred, vertex_1_d3_pred, vertex_1_d4_pred, vertex_1_theta_pred = tf.split(value=vertex_1_y_pred_geo,
                                                                                                 num_or_size_splits=5,
                                                                                                 axis=3)
    vertex_1_area_gt = (vertex_1_d1_gt + vertex_1_d3_gt) * (vertex_1_d2_gt + vertex_1_d4_gt)
    vertex_1_area_pred = (vertex_1_d1_pred + vertex_1_d3_pred) * (vertex_1_d2_pred + vertex_1_d4_pred)
    vertex_1_w_union = tf.minimum(vertex_1_d2_gt, vertex_1_d2_pred) + tf.minimum(vertex_1_d4_gt, vertex_1_d4_pred)
    vertex_1_h_union = tf.minimum(vertex_1_d1_gt, vertex_1_d1_pred) + tf.minimum(vertex_1_d3_gt, vertex_1_d3_pred)
    vertex_1_area_intersect = vertex_1_w_union * vertex_1_h_union
    vertex_1_area_union = vertex_1_area_gt + vertex_1_area_pred - vertex_1_area_intersect
    vertex_1_L_AABB = -tf.log((vertex_1_area_intersect + 1.0) / (vertex_1_area_union + 1.0))
    vertex_1_L_theta = 1 - tf.cos(vertex_1_theta_pred - vertex_1_theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(vertex_1_L_AABB * vertex_1_y_true_cls * vertex_1_training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(vertex_1_L_theta * vertex_1_y_true_cls * vertex_1_training_mask))
    vertex_1_L_g = vertex_1_L_AABB + 20 * vertex_1_L_theta


    # vertex_d1 -> top, d2->right, d3->bottom, d4->left
    vertex_2_d1_gt, vertex_2_d2_gt, vertex_2_d3_gt, vertex_2_d4_gt, vertex_2_theta_gt = tf.split(value=vertex_2_y_true_geo,
                                                                                                 num_or_size_splits=5,
                                                                                                 axis=3)
    vertex_2_d1_pred, vertex_2_d2_pred, vertex_2_d3_pred, vertex_2_d4_pred, vertex_2_theta_pred = tf.split(
        value=vertex_2_y_pred_geo,
        num_or_size_splits=5,
        axis=3)
    vertex_2_area_gt = (vertex_2_d1_gt + vertex_2_d3_gt) * (vertex_2_d2_gt + vertex_2_d4_gt)
    vertex_2_area_pred = (vertex_2_d1_pred + vertex_2_d3_pred) * (vertex_2_d2_pred + vertex_2_d4_pred)
    vertex_2_w_union = tf.minimum(vertex_2_d2_gt, vertex_2_d2_pred) + tf.minimum(vertex_2_d4_gt, vertex_2_d4_pred)
    vertex_2_h_union = tf.minimum(vertex_2_d1_gt, vertex_2_d1_pred) + tf.minimum(vertex_2_d3_gt, vertex_2_d3_pred)
    vertex_2_area_intersect = vertex_2_w_union * vertex_2_h_union
    vertex_2_area_union = vertex_2_area_gt + vertex_2_area_pred - vertex_2_area_intersect
    vertex_2_L_AABB = -tf.log((vertex_2_area_intersect + 1.0) / (vertex_2_area_union + 1.0))
    vertex_2_L_theta = 1 - tf.cos(vertex_2_theta_pred - vertex_2_theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(vertex_2_L_AABB * vertex_2_y_true_cls * vertex_2_training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(vertex_2_L_theta * vertex_2_y_true_cls * vertex_2_training_mask))
    vertex_2_L_g = vertex_2_L_AABB + 20 * vertex_2_L_theta


    # vertex_d1 -> top, d2->right, d3->bottom, d4->left
    vertex_3_d1_gt, vertex_3_d2_gt, vertex_3_d3_gt, vertex_3_d4_gt, vertex_3_theta_gt = tf.split(value=vertex_3_y_true_geo,
                                                                                                 num_or_size_splits=5,
                                                                                                 axis=3)
    vertex_3_d1_pred, vertex_3_d2_pred, vertex_3_d3_pred, vertex_3_d4_pred, vertex_3_theta_pred = tf.split(
        value=vertex_3_y_pred_geo,
        num_or_size_splits=5,
        axis=3)
    vertex_3_area_gt = (vertex_3_d1_gt + vertex_3_d3_gt) * (vertex_3_d2_gt + vertex_3_d4_gt)
    vertex_3_area_pred = (vertex_3_d1_pred + vertex_3_d3_pred) * (vertex_3_d2_pred + vertex_3_d4_pred)
    vertex_3_w_union = tf.minimum(vertex_3_d2_gt, vertex_3_d2_pred) + tf.minimum(vertex_3_d4_gt, vertex_3_d4_pred)
    vertex_3_h_union = tf.minimum(vertex_3_d1_gt, vertex_3_d1_pred) + tf.minimum(vertex_3_d3_gt, vertex_3_d3_pred)
    vertex_3_area_intersect = vertex_3_w_union * vertex_3_h_union
    vertex_3_area_union = vertex_3_area_gt + vertex_3_area_pred - vertex_3_area_intersect
    vertex_3_L_AABB = -tf.log((vertex_3_area_intersect + 1.0) / (vertex_3_area_union + 1.0))
    vertex_3_L_theta = 1 - tf.cos(vertex_3_theta_pred - vertex_3_theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(vertex_3_L_AABB * vertex_3_y_true_cls * vertex_3_training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(vertex_3_L_theta * vertex_3_y_true_cls * vertex_3_training_mask))
    vertex_3_L_g = vertex_3_L_AABB + 20 * vertex_3_L_theta


    # vertex_d1 -> top, d2->right, d3->bottom, d4->left
    vertex_4_d1_gt, vertex_4_d2_gt, vertex_4_d3_gt, vertex_4_d4_gt, vertex_4_theta_gt = tf.split(value=vertex_4_y_true_geo,
                                                                                                 num_or_size_splits=5,
                                                                                                 axis=3)
    vertex_4_d1_pred, vertex_4_d2_pred, vertex_4_d3_pred, vertex_4_d4_pred, vertex_4_theta_pred = tf.split(
        value=vertex_4_y_pred_geo,
        num_or_size_splits=5,
        axis=3)
    vertex_4_area_gt = (vertex_4_d1_gt + vertex_4_d3_gt) * (vertex_4_d2_gt + vertex_4_d4_gt)
    vertex_4_area_pred = (vertex_4_d1_pred + vertex_4_d3_pred) * (vertex_4_d2_pred + vertex_4_d4_pred)
    vertex_4_w_union = tf.minimum(vertex_4_d2_gt, vertex_4_d2_pred) + tf.minimum(vertex_4_d4_gt, vertex_4_d4_pred)
    vertex_4_h_union = tf.minimum(vertex_4_d1_gt, vertex_4_d1_pred) + tf.minimum(vertex_4_d3_gt, vertex_4_d3_pred)
    vertex_4_area_intersect = vertex_4_w_union * vertex_4_h_union
    vertex_4_area_union = vertex_4_area_gt + vertex_4_area_pred - vertex_4_area_intersect
    vertex_4_L_AABB = -tf.log((vertex_4_area_intersect + 1.0) / (vertex_4_area_union + 1.0))
    vertex_4_L_theta = 1 - tf.cos(vertex_4_theta_pred - vertex_4_theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(vertex_4_L_AABB * vertex_4_y_true_cls * vertex_4_training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(vertex_4_L_theta * vertex_4_y_true_cls * vertex_4_training_mask))
    vertex_4_L_g = vertex_4_L_AABB + 20 * vertex_4_L_theta

    vertex_loss_1 = tf.reduce_mean(vertex_1_L_g * vertex_1_y_true_cls * vertex_1_training_mask) + vertex_1_classification_loss
    vertex_loss_2 = tf.reduce_mean(vertex_2_L_g * vertex_2_y_true_cls * vertex_2_training_mask) + vertex_2_classification_loss
    vertex_loss_3 = tf.reduce_mean(
        vertex_3_L_g * vertex_3_y_true_cls * vertex_3_training_mask) + vertex_3_classification_loss
    vertex_loss_4 = tf.reduce_mean(
        vertex_4_L_g * vertex_4_y_true_cls * vertex_4_training_mask) + vertex_4_classification_loss

    vertex_loss = (vertex_loss_1+vertex_loss_2+vertex_loss_3+vertex_loss_4)/4

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss + tf.reduce_mean(vertex_L_g * vertex_y_true_cls * vertex_training_mask) + vertex_classification_loss + vertex_loss
    # return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss