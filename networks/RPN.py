from keras.layers import Conv2D


def rpn(feature_extractor_layers, total_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(feature_extractor_layers)

    x_class_out = Conv2D(total_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regressor_out = Conv2D(total_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class_out, x_regressor_out, feature_extractor_layers]


