import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms

import model
from data import restore_rectangle
from perspective_trans import transform
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class EASTDetection(object):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        self.checkpoint_path = '/home/huluwa/Etranform/model_old/'
        self.session = tf.Session(config=config)
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        self.f_score, self.f_geometry, self.vertex_score, self.vertex_geometry,\
        self.vertex_1_score, self.vertex_1_geometry, \
        self.vertex_2_score, self.vertex_2_geometry, \
        self.vertex_3_score, self.vertex_3_geometry, \
        self.vertex_4_score, self.vertex_4_geometry = model_4v.model(self.input_images, is_training=False)

        self.variable_averages = tf.train.ExponentialMovingAverage(0.997, self.global_step)
        self.saver = tf.train.Saver(self.variable_averages.variables_to_restore())
        self.ckpt_state = tf.train.get_checkpoint_state(self.checkpoint_path)
        self.model_path = os.path.join(self.checkpoint_path, os.path.basename(self.ckpt_state.model_checkpoint_path))
        self.saver.restore(self.session,self.model_path)

    def table_detection(self, image, image_color):
        img_e = np.expand_dims(image, axis=2)
        img_e_c = np.concatenate((img_e, img_e, img_e), axis=-1)
        im_resized, (ratio_h, ratio_w) = resize_image(img_e_c)
        score, geometry = self.session.run([self.f_score, self.f_geometry], feed_dict={self.input_images: [im_resized]})
        score_0 = score[0]
        score_map = cv2.resize(score_0, dsize=None, fx=1 / ratio_w, fy=1 / ratio_h, interpolation=cv2.INTER_AREA)

        return score_map


    def main_detection(self, images):
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        with tf.device('/gpu:2'):
            total_boxes = []
            total_image = []
            for image in images:
                im = image[:,:,::-1]
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                score, geometry, vertex_score,  vertex_geometry, vertex_1_score,  vertex_1_geometry, vertex_2_score,  vertex_2_geometry, vertex_3_score,  vertex_3_geometry, vertex_4_score,  vertex_4_geometry= self.session.run([self.f_score, self.f_geometry, self.vertex_score, self.vertex_geometry, self.vertex_1_score, self.vertex_1_geometry, self.vertex_2_score, self.vertex_2_geometry, self.vertex_3_score, self.vertex_3_geometry, self.vertex_4_score, self.vertex_4_geometry], feed_dict={self.input_images: [im_resized]})
                need_points = []
                vertex_1_boxes = detect(score_map=vertex_1_score, geo_map=vertex_1_geometry)
                print('vertex_1_boxes:', len(vertex_1_boxes))
                if vertex_1_boxes != []:

                    vertex_1_boxes = vertex_1_boxes[:, :8].reshape((-1, 4, 2))
                    vertex_1_boxes[:, :, 0] /= ratio_w
                    vertex_1_boxes[:, :, 1] /= ratio_h
                    need_points.append(vertex_1_boxes[0][0])
                for box in vertex_1_boxes:
                    cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 155, 155),
                                  thickness=3)

                vertex_2_boxes = detect(score_map=vertex_2_score, geo_map=vertex_2_geometry)
                print('vertex_2_boxes:', len(vertex_2_boxes))
                if vertex_2_boxes != []:

                    vertex_2_boxes = vertex_2_boxes[:, :8].reshape((-1, 4, 2))
                    vertex_2_boxes[:, :, 0] /= ratio_w
                    vertex_2_boxes[:, :, 1] /= ratio_h
                    need_points.append(vertex_2_boxes[0][1])
                for box in vertex_2_boxes:
                    cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255),
                                  thickness=3)

                vertex_3_boxes = detect(score_map=vertex_3_score, geo_map=vertex_3_geometry)
                print('vertex_3_boxes:', len(vertex_3_boxes))
                if vertex_3_boxes != []:

                    vertex_3_boxes = vertex_3_boxes[:, :8].reshape((-1, 4, 2))
                    vertex_3_boxes[:, :, 0] /= ratio_w
                    vertex_3_boxes[:, :, 1] /= ratio_h
                    need_points.append(vertex_3_boxes[0][2])
                for box in vertex_3_boxes:
                    cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 0, 0),
                                  thickness=3)

                vertex_4_boxes = detect(score_map=vertex_4_score, geo_map=vertex_4_geometry)
                print('vertex_4_boxes:', len(vertex_4_boxes))
                if vertex_4_boxes != []:

                    vertex_4_boxes = vertex_4_boxes[:, :8].reshape((-1, 4, 2))
                    vertex_4_boxes[:, :, 0] /= ratio_w
                    vertex_4_boxes[:, :, 1] /= ratio_h
                    need_points.append(vertex_4_boxes[0][3])
                for box in vertex_4_boxes:
                    cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                                  thickness=3)

                if len(need_points) == 4:
                    print(need_points)
                    dst_im = transform(need_points, im)
                    total_image.append(dst_im)
            return total_boxes, total_image


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, score_map_thresh=0.5, box_thresh=0.005, nms_thres=0.01):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)


    if boxes.shape[0] == 0:
        return []

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


instance = EASTDetection()
if __name__ == '__main__':

    import time
    root_dir = '/home/huluwa/Etranform/ver_data/image/'
    save_dir = '/home/huluwa/Etranform/ver_data/result/'
    image_name_list = os.listdir(root_dir)

    for line in image_name_list:
        print(line)
        start = time.time()
        image_path = os.path.join(root_dir, line)
        image = cv2.imread(image_path)
        image_path = os.path.join(root_dir, line)
        print(image_path)
        # cv2.imshow('1',image)
        # cv2.waitKey(0)
        boxes, images = instance.main_detection([image])
        save_path = os.path.join(save_dir, line)
        end = time.time()
        print(save_path)
        print('used time :',end-start)
        # if len(images)>0:
        cv2.imwrite(save_path, images[0])


