from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import get_file

def identity_block(input_tensor, filters):
    filter1, filter2, filter3 = filters
    shortcut = input_tensor
    
    # First component
    x = Conv2D(filter1, (1, 1), padding="valid", kernel_initializer=glorot_uniform(seed=0))(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # Second component
    x = Conv2D(filter2, (3, 3), padding="same", kernel_initializer=glorot_uniform(seed=0))(x) 
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # Third component
    x = Conv2D(filter3, (1, 1), padding="valid", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    
    return x

def convolutional_block(input_tensor, filters, stride):
    filter1, filter2, filter3 = filters
    shortcut = input_tensor
    
    # First component
    x = Conv2D(filter1, (1, 1), strides=(stride, stride), padding="valid", kernel_initializer=glorot_uniform(seed=0))(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # Second component
    x = Conv2D(filter2, (3, 3), padding="same", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # Third component
    x = Conv2D(filter3, (1, 1), padding="valid", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    
    shortcut = Conv2D(filter3, (1, 1), strides=(stride, stride), padding="valid", kernel_initializer=glorot_uniform(seed=0))(shortcut)
    shortcut = BatchNormalization(axis=3)(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def ResNet50(input_shape=(224, 224, 3), weights=None, num_classes=1000, include_top=True):
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    input_tensor = Input(input_shape)
    x = ZeroPadding2D((3, 3))(input_tensor)
    
    # Stage 1
    x = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Stage 2
    x = convolutional_block(x, [64, 64, 256], 1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])
    
    # Stage 3
    x = convolutional_block(x, [128, 128, 512], 2)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    
    # Stage 4
    x = convolutional_block(x, [256, 256, 1024], 2)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    
    # Stage 5
    x = convolutional_block(x, [512, 512, 2048], 2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    
    if include_top:
        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(x)
    
    model = Model(inputs=input_tensor, outputs=x)
    
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
            
        model.load_weights(weights_path)
    
    return model
