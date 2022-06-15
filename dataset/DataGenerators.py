from __future__ import absolute_import
import numpy as np
import cv2
import random
from dataset import augmentation
import threading
import itertools

from dataset.utils import get_new_img_size


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


class SampleSelector:
    def __init__(self, class_count):
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):

        class_in_img = False

        for bbox in img_data['bboxes']:
            cls_name = bbox['class']
            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break

        if class_in_img:
            return False
        else:
            return True


def calc_rpn(C, image_data, width, height, resized_width, resized_height, img_length_calc_function):
    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchor_ratios = len(anchor_ratios)

    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors)).astype(int)
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors)).astype(int)
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(image_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4))
    best_dx_for_bbox = np.zeros((num_bboxes, 4))

    gt_box = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(image_data['bboxes']):
        gt_box[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gt_box[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gt_box[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gt_box[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchor_ratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):

                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    bbox_type = 'neg'

                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        curr_iou = iou([gt_box[bbox_num, 0], gt_box[bbox_num, 2], gt_box[bbox_num, 1], gt_box[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gt_box[bbox_num, 0] + gt_box[bbox_num, 1]) / 2.0
                            cy = (gt_box[bbox_num, 2] + gt_box[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = 1.0 * (gt_box[bbox_num, 1] - gt_box[bbox_num, 0]) / (x2_anc - x1_anc)
                            th = 1.0 * (gt_box[bbox_num, 3] - gt_box[bbox_num, 2]) / (y2_anc - y1_anc)

                        if image_data['bboxes'][bbox_num]['class'] != 'bg':

                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchor_ratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 2] = best_regr[0:2]
                        y_rpn_regr[jy, ix, start + 2:start + 4] = np.log(best_regr[2:])

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                idx, 2] + n_anchor_ratios *
                           best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                idx, 2] + n_anchor_ratios *
                          best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchor_ratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 2] = best_dx_for_bbox[
                                                                                                      idx, 0:2]
            y_rpn_regr[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start + 2:start + 4] = np.log(
                best_dx_for_bbox[idx, 2:4])

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class ThreadSafeIter:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            np.random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:

                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                if mode == 'train':
                    img_data_aug, x_image = augmentation.augment(img_data, C, do_augment=True)
                else:
                    img_data_aug, x_image = augmentation.augment(img_data, C, do_augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_image.shape

                assert cols == width
                assert rows == height

                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                x_image = cv2.resize(x_image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height,
                                                     img_length_calc_function)
                except Exception as error:
                    print({f'Error: {str(error)}'})
                    continue

                x_image = x_image[:, :, (2, 1, 0)]  # BGR -> RGB
                x_image = x_image.astype(np.float32)
                x_image[:, :, 0] -= C.img_channel_mean[0]
                x_image[:, :, 1] -= C.img_channel_mean[1]
                x_image[:, :, 2] -= C.img_channel_mean[2]
                x_image /= C.img_scaling_factor

                x_image = np.transpose(x_image, (2, 0, 1))
                x_image = np.expand_dims(x_image, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling

                if backend == 'tf':
                    x_image = np.transpose(x_image, (0, 2, 3, 1))
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_image), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as error:
                print({f'Error: {str(error)}'})
                continue
