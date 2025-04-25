




**Data**
Data folder contains metadata, datasets imported in Notebooks fromt the repositry, sample images and information realted to datsets.


# A Deep Learning Based Feature Fusion Method for Automatic Classification of Malaria Parasite using Blood Smear Images

This robust malaria disease classification model from blood smear images is built using Convolutional Neural Networks (CNN). Leveraging powerful Python libraries such as TensorFlow, Keras, PIL, and scikit-learn, the model developed is capable of accurately identifying parasites (infections) in blood smear images. This comprehensive approach includes preprocessing with NumPy and pandas, data visualization with Matplotlib and Seaborn, and efficient model training utilizing advanced CNN layers like Conv2D, MaxPooling2D, Flatten, Dropout, and Dense layers. 

Explore the intricacies of my approach, from data augmentation to early stopping, as I aim for precise diagnosis through cutting-edge image classification models based on deep learning techniques.

## Convolutional Neural Networks.

A CNN is a type of neural network that's really good at looking at pictures and figuring out what's in them. This specific model is set up to look at images that are 64x64 pixels with 3 color channels (red, green, blue) and decide between categories.

![image](https://github.com/hari255/Neural-Networks/assets/59302293/6be05b5d-6bf4-4a45-b2dd-fb604d538060)

## Prerequisites.

Keras 2.2.0

Tensorflow-GPU 1.9.0

Scikit-Learn

PIL

Matplotlib and Seaborn

## Data Exploration

**Data set size:** 27558 samples of images, which include parasites and non-parasite cells. The image below is plotted using Python on the dataset.

<img width="460" alt="image" src="https://github.com/hari255/Neural-Networks/assets/59302293/247ea202-56a1-45a4-a959-bec0e40e9bee">

+ Infected/malaria parasitized cells have some form of disturbances within the cell, with a dark color formation.

+ Uninfected cells have a uniform color throughout the image.

To accurately identify parasites in cell images and train our model to distinguish them, we must focus on the key aspects that differentiate these categories. This involves analyzing and understanding the unique characteristics and features that set each parasite apart. By doing so, we can ensure that our model is well-equipped to recognize and classify the various types of parasites accurately.


+ **Average Uninfected Image**
<img width="389" alt="image" src="https://github.com/hari255/Neural-Networks/assets/59302293/992a7520-26a7-419e-84b3-ce3601fc4f64">

+ **Average Infected Image**
<img width="392" alt="image" src="https://github.com/hari255/Neural-Networks/assets/59302293/e950ebaf-1cfe-4b4a-8e4f-14bf80222e60">


**The mean images for both parasitized and uninfected are pretty much the same because the difference between these two images is very small (infection is the only difference). The average image is obviously the larger part of it, and it seems most likely is the idea.**

## Data Transformation

In the data transformation stage, I've tried to use multiple techniques used in image processing and Computer Vision and experimenting with my dataset. These techniques are useful in different stages of model building.

| Technique | Description |
| ------ | ----------- |
| RGB to HSV  | To separate image brightness from color information |
| Gaussian smoothing | To remove noise from the image |
| Data Augmentation   | To slightly alter the image and generate new one |


## Converting images from RGB to HSV using OpenCV

The purpose of converting RGB images to HSV (Hue, Saturation, Value) using OpenCV is to facilitate more effective image processing and analysis. The HSV color space separates image intensity (brightness) from color information, which can be particularly useful for various image processing tasks. 

In the HSV color space, it's easy to separate colors based on their hue. This is useful for segmenting objects in an image based on color. This property helps us identify and differentiate the infected cell images.

Plotted below image after converting from RGB to HSV, it is quite helpful in our problem to efficiently identify the parasite in HSV format.
![image](https://github.com/hari255/Neural-Networks/assets/59302293/f5ef2d2c-aa38-44c1-b787-5f5ad3a4c2fb)


## Gaussian Blurring

**Gaussian Blurring or Smoothing is a technique that helps in removing noise from an image. It uses a Gaussian kernel to weigh the neighboring pixels based on a Gaussian distribution.**

<img width="252" alt="image" src="https://github.com/hari255/Neural-Networks/assets/59302293/1165c5af-51d5-4173-950e-cc82a864bbb5">

+ (x,y) are the coordinates of the pixel.
+ σ is the standard deviation of the Gaussian distribution, which controls the amount of blurring.

After Gaussian smoothing, the images look like below

![image](https://github.com/hari255/Neural-Networks/assets/59302293/f20b57b9-ed30-4b57-89d7-f6d7ce55cf38)


## Model 

**Considering the problem of detecting parasites, our model needs to have a High `True positive` rate and a low `false negative` rate.  If we manage to build such a model, then it would be appropriate for malaria classification. It is because we can't take any chances on the `False Negatives`, which means predicting uninfected when actually infected, which can lead to serious issues.**


`Python code for the model Architecture`
``` py
from tensorflow.keras.layers import  BatchNormalization
from keras import layers
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.models import Model
from keras.regularizers import l2


def conv2d_bn(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay)
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides)
    layer = Activation('relu')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, weight_decay, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    return out


def ResNet18(classes, input_shape, weight_decay=1e-4):
    input = Input(shape=input_shape)
    x = input
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='sigmoid')(x)
    model = Model(input, x, name='ResNet18')
    return model

from keras import losses
from keras import optimizers

weight_decay = 1e-4
lr = 1e-1
num_classes = 1
resnet18 = ResNet18(input_shape=(110, 110, 3), classes=num_classes, weight_decay=weight_decay)
opt = optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
resnet18.compile(optimizer=opt,
                 loss=losses.binary_crossentropy,
                 metrics=['accuracy'])
resnet18.summary()

# Training the model on the input data by using the fit_generator function 
history = resnet18.fit_generator(train_generator, steps_per_epoch = total_train // batch_size, 
                       epochs = epochs, 
                       validation_data = validation_generator, 
                       validation_steps = total_val // batch_size) 

```
     
---

To benchmark my innovative approach, I've conducted a comprehensive performance analysis against the renowned VGG16 image-detection model. The findings not only highlight the superiority of my model but also shed light on key metrics that set it apart.
VGG16 model; https://keras.io/api/applications/vgg/ 

Team, Keras. (2016). Keras Documentation: VGG16 and VGG19. https://keras.io/api/applications/vgg/

## Pre-trained CNN models

We also evaluated the performance of pre-trained CNN models including VGG-16 and ResNet-50V2 towards extracting the features from the parasitized and uninfected cells. The feature extraction layers of these models were frozen, while the dense (fully connected) layers were retrained on the malaria cell image dataset. The pre-trained models could not perform well than the proposed feature fusion model in terms of precision, recall, f1 and accuracy scores.


---------------------------------------------------------
---------------------------------------------------------
