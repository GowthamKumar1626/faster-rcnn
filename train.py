from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os

import config
from dataset.parser import get_data
from dataset import DataGenerators
from networks import vgg16 as nn
from networks import RPN
from networks import classifier
from losses import loss as loss_functions
from utils import helpers

from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils

if 'tensorflow' == K.backend():
    import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))

sys.setrecursionlimit(40000)

# Parsers
parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=4)
parser.add_option("--feature_extractor", dest="network", help="Base network to use. Supports vgg or resnet50.",
                  default='vgg')
parser.add_option("--n_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=5)
parser.add_option("--cf", dest="config_filename",
                  help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
                  default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.",
                  default=None)
parser.add_option("--opt", dest="optimizers", help="set the optimizer to use", default="SGD")
parser.add_option("--lr", dest="lr", help="learn rate", type=float, default=1e-3)
parser.add_option("--load", dest="load", help="What model to load", default=None)

(options, args) = parser.parse_args()

if not options.train_path:
    parser.error('Error: path to training data must be specified. Pass --path to command line')

# Initializing Config variable
C = config.Config()

C.use_horizontal_flips = True
C.use_vertical_flips = False
C.rot_90 = False

# Configuring pretrained models
if not os.path.isdir("models"):
    os.mkdir("models")
if not os.path.isdir("models/" + options.network):
    os.mkdir(os.path.join("models", options.network))
C.model_path = os.path.join("models", options.network, options.dataset + ".hdf5")
C.num_rois = int(options.num_rois)

if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    C.base_net_weights = nn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(options.train_path, options.cat)

# Checking for labels categorized as labels
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

pprint.pprint(classes_count)

# Configuring config files
config_output_filename = options.config_filename
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

# Shuffling dataset
random.shuffle(all_imgs)

# Splitting dataset
num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = DataGenerators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length,
                                              K.image_dim_ordering(), mode='train')
data_gen_val = DataGenerators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,
                                            K.image_dim_ordering(), mode='val')

# Image input shapes
if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

shared_layers = nn.feature_extractor(img_input, trainable=True)

# RPN
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = RPN.rpn(shared_layers, num_anchors)

# Classifier
classifier = classifier.classifier(shared_layers, roi_input, C.num_rois, total_classes=len(classes_count),
                                   trainable=True)

# RPN MODEL
model_rpn = Model(img_input, rpn[:2])
# Classifier Model
model_classifier = Model([img_input, roi_input], classifier)

# Model that holds both the RPN and the Classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# Loading pretrained weights
try:
    print('Loading pre-trained weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except Exception as error:
    print(f'{"details": {str(error)}}')
    print('Could not load pretrained model weights')

# Configuring optimizers
if options.optimizers == "SGD":
    if options.rpn_weight_path is not None:
        optimizer = SGD(lr=options.lr / 100, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr / 5, decay=0.0005, momentum=0.9)
    else:
        optimizer = SGD(lr=options.lr / 10, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr / 10, decay=0.0005, momentum=0.9)
else:
    optimizer = Adam(lr=options.lr, clipnorm=0.001)
    optimizer_classifier = Adam(lr=options.lr, clipnorm=0.001)

if options.load is not None:
    print("Loading previous model from ", options.load)
    model_rpn.load_weights(options.load, by_name=True)
    model_classifier.load_weights(options.load, by_name=True)
elif options.rpn_weight_path is not None:
    print("Loading RPN weights from ", options.rpn_weight_path)
    model_rpn.load_weights(options.rpn_weight_path, by_name=True)
else:
    print("There are no previous model was loaded")

# Compile models
# Compiling RPN Model
model_rpn.compile(
    optimizer=optimizer,
    loss=[loss_functions.rpn_loss_cls(num_anchors),
          loss_functions.rpn_loss_regr(num_anchors)]
)

# Compiling Classifier Model
model_classifier.compile(
    optimizer=optimizer_classifier,
    loss=[loss_functions.class_loss_cls, loss_functions.class_loss_regr(len(classes_count) - 1)],
    metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'}
)

# Compiling Combined model
model_all.compile(optimizer='sgd', loss='mae')

# Training configurations

epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    if epoch_num < 3 and options.rpn_weight_path is not None:
        K.set_value(model_rpn.optimizer.lr, options.lr / 30)
        K.set_value(model_classifier.optimizer.lr, options.lr / 3)

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            X, Y, img_data = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)
            R = helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.4,
                                   max_boxes=300)
            X2, Y1, Y2, IouS = helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except Exception as error:
                    print(f'{"details": {str(error)}}')
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=True).tolist()
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3])),
                            ("average number of objects", len(selected_pos_samples))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)

                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue
