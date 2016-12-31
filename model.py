import argparse

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, MaxPooling3D, ELU
from keras.models import Sequential, model_from_json
from sklearn.model_selection import train_test_split
import json



# This method is for displaying the image from a file for debugging purpose
def getImage(file1, debug=False, plot=False):
    if (debug):
        print(file1)
    img = cv2.imread(file1)
    correctedimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (plot):
        plotImage(correctedimg, debug)

    return correctedimg


# This method is for displaying the image from a file for debugging purpose
def plotImage(image, debug=False, title=""):
    if (debug):
        print(image.shape)
        print(image)
    plt.imshow(image)
    plt.figtext(0, 0, title)
    plt.show()


# Simple normalization , dividing the data by 255 and setting the mean to 0
def normalizeImage(image):
    image = image / 255.
    image -= 0.5
    return image


def getnormalizeddata(filelocations, debug=False):
    xnormalized = []
    for filename in filelocations:
        image = getImage(filename)
        resize = resizeimage(image, 8)
        normalizedimage = normalizeImage(resize)
        xnormalized.append(normalizedimage)
        if debug and len(xnormalized) % 100 == 0:
            print("Loaded " + str(len(xnormalized)) + "images")

    ydata = yoriginal
    return xnormalized, ydata


def resizeimage(image, factor):
    return cv2.resize(image, (int(image.shape[0] / factor), int(image.shape[1] / factor)))


def flipimage(image):
    flippedimage = cv2.flip(image, 1)
    return flippedimage


def augmentdata(xnormalizeddata, ydata):
    datalen = len(ydata)
    i = 0
    while i < datalen:
        currentx = xnormalizeddata[i]
        flippedimage = flipimage(currentx)
        xnormalizeddata.append(flippedimage)
        ydata = ydata.append(pd.Series(ydata[i] * -1))
        i = i + 1
    return xnormalizeddata, ydata


def getFinalData(imagefiles, augment=True):
    xnormalized, ydata = getnormalizeddata(imagefiles, debug=True)
    if (augment):
        xfinal, yfinal = augmentdata(xnormalized, ydata)
    else:
        xfinal = xnormalized
        yfinal = ydata
    return np.asarray(xfinal), yfinal


def generate_data(xdata,ydata,batch_size=32):
    while 1:

        yield xdata,ydata



def trainmodel(X_train, X_valid, y_train, y_valid, batch_size=128, nb_epoch=20, kernel_size=(5, 5), nb_filters=32,model=None):
    if model is None:
        model = Sequential()
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='same',
                                input_shape=(40, 20, 3,)))
        model.add(MaxPooling2D())
        model.add(Convolution2D(nb_filters * 2, kernel_size[0], kernel_size[1],
                                border_mode='same',
                                input_shape=(20, 10, 32,)))
        model.add(MaxPooling2D())
        model.add(Convolution2D(nb_filters * 4, kernel_size[0], kernel_size[1],
                                border_mode='same',
                                input_shape=(20, 10, 64,)))
        model.add(MaxPooling2D())
        model.add(Convolution2D(nb_filters * 8, kernel_size[0], kernel_size[1],
                                border_mode='same',
                                input_shape=(10, 5, 64,)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        print("Loaded model from file")

    print(model.summary())
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, nb_epoch=nb_epoch)
    return model


def save_model(model, filename="model.h5",model_arch="model.json"):
    model.save(filepath=filename)
    model_json = model.to_json()
    with open(model_arch, 'w') as f:
        json.dump(model_json, f)
    print("Saved model to filename="+filename + ", and model_arch="+model_arch)




if __name__ == '__main__':
    dataframe = pd.read_csv("driving_log.csv", delim_whitespace=False, header=None)
    centerimages = (dataframe[:][0])
    yoriginal = (dataframe[:][3])
    xfinal, yfinal = getFinalData(centerimages)

    X_train, X_valid, y_train, y_valid = train_test_split(
        xfinal,
        yfinal,
        test_size=0.05,
        random_state=832289)

    model = None
    with open('model.json', 'r') as jfile:
        if jfile is not None:
            model = model_from_json(json.load(jfile))
            model.compile("adam", "mse")
            weights_file = "model.h5"
            model.load_weights(weights_file)
            print("Loading model from file")

    model = trainmodel(X_train, X_valid, y_train, y_valid, nb_epoch=10,model=model)
    save_model(model)

