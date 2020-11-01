import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import cv2


tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 14, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/huluwa/Etranform/model/ver/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', '/home/huluwa/Etranform/model/resnet_v1_50.ckpt', '')

import model
import data

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))


def tower_loss(images, score_maps, geo_maps, training_masks, vertex_score_maps, vertex_geo_maps, vertex_training_masks, vertex_1_score_maps, vertex_1_geo_maps, vertex_1_training_masks, vertex_2_score_maps, vertex_2_geo_maps, vertex_2_training_masks, vertex_3_score_maps, vertex_3_geo_maps, vertex_3_training_masks, vertex_4_score_maps, vertex_4_geo_maps, vertex_4_training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry, vertex_f_score, vertex_f_geometry, vertex_1_f_score, vertex_1_f_geometry, vertex_2_f_score, vertex_2_f_geometry, vertex_3_f_score, vertex_3_f_geometry, vertex_4_f_score, vertex_4_f_geometry = model_4v.model(images, is_training=True)

    model_loss = model_4v.loss(score_maps, f_score,
                            geo_maps, f_geometry,
                            training_masks,
                               vertex_score_maps, vertex_f_score, vertex_geo_maps, vertex_f_geometry,
                               vertex_training_masks,
                               vertex_1_score_maps, vertex_1_f_score, vertex_1_geo_maps, vertex_1_f_geometry,
                               vertex_1_training_masks,
                               vertex_2_score_maps, vertex_2_f_score, vertex_2_geo_maps, vertex_2_f_geometry,
                               vertex_2_training_masks,
                               vertex_3_score_maps, vertex_3_f_score, vertex_3_geo_maps, vertex_3_f_geometry,
                               vertex_3_training_masks,
                               vertex_4_score_maps, vertex_4_f_score, vertex_4_geo_maps, vertex_4_f_geometry,
                               vertex_4_training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    vertex_input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_input_score_maps')
    if FLAGS.geometry == 'RBOX':
        vertex_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='vertex_input_geo_maps')
    else:
        vertex_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='vertex_input_geo_maps')
    vertex_input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_input_training_masks')

    vertex_1_input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_1_input_score_maps')
    if FLAGS.geometry == 'RBOX':
        vertex_1_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='vertex_1_input_geo_maps')
    else:
        vertex_1_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='vertex_1_input_geo_maps')
    vertex_1_input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_1_input_training_masks')

    vertex_2_input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_2_input_score_maps')
    if FLAGS.geometry == 'RBOX':
        vertex_2_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='vertex_2_input_geo_maps')
    else:
        vertex_2_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='vertex_2_input_geo_maps')
    vertex_2_input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_2_input_training_masks')

    vertex_3_input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_3_input_score_maps')
    if FLAGS.geometry == 'RBOX':
        vertex_3_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='vertex_3_input_geo_maps')
    else:
        vertex_3_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='vertex_3_input_geo_maps')
    vertex_3_input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_3_input_training_masks')

    vertex_4_input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_4_input_score_maps')
    if FLAGS.geometry == 'RBOX':
        vertex_4_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='vertex_4_input_geo_maps')
    else:
        vertex_4_input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='vertex_4_input_geo_maps')
    vertex_4_input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vertex_4_input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)


    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    vertex_1_input_score_maps_split = tf.split(vertex_1_input_score_maps, len(gpus))
    vertex_1_input_geo_maps_split = tf.split(vertex_1_input_geo_maps, len(gpus))
    vertex_1_input_training_masks_split = tf.split(vertex_1_input_training_masks, len(gpus))

    vertex_2_input_score_maps_split = tf.split(vertex_2_input_score_maps, len(gpus))
    vertex_2_input_geo_maps_split = tf.split(vertex_2_input_geo_maps, len(gpus))
    vertex_2_input_training_masks_split = tf.split(vertex_2_input_training_masks, len(gpus))

    vertex_3_input_score_maps_split = tf.split(vertex_3_input_score_maps, len(gpus))
    vertex_3_input_geo_maps_split = tf.split(vertex_3_input_geo_maps, len(gpus))
    vertex_3_input_training_masks_split = tf.split(vertex_3_input_training_masks, len(gpus))

    vertex_4_input_score_maps_split = tf.split(vertex_4_input_score_maps, len(gpus))
    vertex_4_input_geo_maps_split = tf.split(vertex_4_input_geo_maps, len(gpus))
    vertex_4_input_training_masks_split = tf.split(vertex_4_input_training_masks, len(gpus))

    vertex_input_score_maps_split = tf.split(vertex_input_score_maps, len(gpus))
    vertex_input_geo_maps_split = tf.split(vertex_input_geo_maps, len(gpus))
    vertex_input_training_masks_split = tf.split(vertex_input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                vertex_isms = vertex_input_score_maps_split[i]
                vertex_igms = vertex_input_geo_maps_split[i]
                vertex_itms = vertex_input_training_masks_split[i]

                vertex_1_isms = vertex_1_input_score_maps_split[i]
                vertex_1_igms = vertex_1_input_geo_maps_split[i]
                vertex_1_itms = vertex_1_input_training_masks_split[i]

                vertex_2_isms = vertex_2_input_score_maps_split[i]
                vertex_2_igms = vertex_2_input_geo_maps_split[i]
                vertex_2_itms = vertex_2_input_training_masks_split[i]

                vertex_3_isms = vertex_3_input_score_maps_split[i]
                vertex_3_igms = vertex_3_input_geo_maps_split[i]
                vertex_3_itms = vertex_3_input_training_masks_split[i]

                vertex_4_isms = vertex_4_input_score_maps_split[i]
                vertex_4_igms = vertex_4_input_geo_maps_split[i]
                vertex_4_itms = vertex_4_input_training_masks_split[i]

                total_loss, model_loss = tower_loss(iis, isms, igms, itms, vertex_isms, vertex_igms, vertex_itms, vertex_1_isms, vertex_1_igms, vertex_1_itms, vertex_2_isms, vertex_2_igms, vertex_2_itms, vertex_3_isms, vertex_3_igms, vertex_3_itms, vertex_4_isms, vertex_4_igms, vertex_4_itms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = icdar_4v.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=512,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))


        start = time.time()
        for step in range(FLAGS.max_steps):
            data = next(data_generator)

            # input_list = []
            # for input_image in data[0]:
                # input_image = cv2.resize(input_image,(1024,1024),interpolation=cv2.INTER_LINEAR)
                # input_list.append(input_image)
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_score_maps: data[2],
                                                                                input_geo_maps: data[3],
                                                                                input_training_masks: data[4],
                                                                                vertex_input_score_maps:data[5],
                                                                                vertex_input_geo_maps:data[6],
                                                                                vertex_input_training_masks:data[7],
                                                                                vertex_1_input_score_maps: data[8],
                                                                                vertex_1_input_geo_maps: data[9],
                                                                                vertex_1_input_training_masks: data[10],
                                                                                vertex_2_input_score_maps: data[11],
                                                                                vertex_2_input_geo_maps: data[12],
                                                                                vertex_2_input_training_masks: data[13],
                                                                                vertex_3_input_score_maps: data[14],
                                                                                vertex_3_input_geo_maps: data[15],
                                                                                vertex_3_input_training_masks: data[16],
                                                                                vertex_4_input_score_maps: data[17],
                                                                                vertex_4_input_geo_maps: data[18],
                                                                                vertex_4_input_training_masks: data[19]
                                                                                })
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus))/(time.time() - start)
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                            input_score_maps: data[2],
                                                                                            input_geo_maps: data[3],
                                                                                            input_training_masks: data[4],
                                                                                            vertex_input_score_maps:data[5],
                                                                                            vertex_input_geo_maps:data[6],
                                                                                            vertex_input_training_masks:data[7],
                                                                                            vertex_1_input_score_maps: data[8],
                                                                                            vertex_1_input_geo_maps: data[9],
                                                                                            vertex_1_input_training_masks: data[10],
                                                                                            vertex_2_input_score_maps: data[11],
                                                                                            vertex_2_input_geo_maps: data[12],
                                                                                            vertex_2_input_training_masks: data[13],
                                                                                            vertex_3_input_score_maps: data[14],
                                                                                            vertex_3_input_geo_maps: data[15],
                                                                                            vertex_3_input_training_masks: data[16],
                                                                                            vertex_4_input_score_maps: data[17],
                                                                                            vertex_4_input_geo_maps: data[18],
                                                                                            vertex_4_input_training_masks: data[19]
                                                                                            })
                summary_writer.add_summary(summary_str, global_step=step)

if __name__ == '__main__':
    tf.app.run()
