from keras import backend as K
from keras.layers import Flatten, Dense, Dropout
from keras.layers import TimeDistributed
from layers.RoiPoolingConvLayer import RoiPoolingConv


def classifier(feature_extractor_layers, input_rois, num_rois, total_classes=21, trainable=False):

    if K.backend() == 'tensorflow':
        pooling_regions = 7
    elif K.backend() == 'theano':
        pooling_regions = 7

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([feature_extractor_layers, input_rois])

    output = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    output = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(output)
    output = TimeDistributed(Dropout(0.5))(output)
    output = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(output)
    output = TimeDistributed(Dropout(0.5))(output)

    out_class = TimeDistributed(Dense(total_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(total_classes))(output)
    out_regressor = TimeDistributed(Dense(4 * (total_classes - 1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(total_classes))(output)

    return [out_class, out_regressor]