import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime

from .my_modules import *

##############################################################################
# NN 기본 형태


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train= x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation='relu', input_shape=(784, )),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# cp_path = "tensorflow_data/basic_model.ckpt"
# cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
model_path = "tensorflow_data/basic_model.h5"


# log_path = "tensorflow_data/log/"
# log_path = os.path.join("log")
tensorboard_callback = keras.callbacks.TensorBoard(profile_batch=1000000)

model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])


model.evaluate(x_test, y_test, verbose=2)
model.save(model_path)

new_model = keras.models.load_model(model_path)
new_model.evaluate(x_test, y_test, verbose=2)

new_model.predict(x_test[0].reshape(-1,784))



##############################################################################
# CNN 기본 형태


import tensorflow as tf
from tensorflow import keras
import os

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# log_path = os.path.join("CNN_logs")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_path, profile_batch=1000000)


# model.fit(train_images, train_labels, epochs=5, callbacks=[tensorboard_callback])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels, verbose=2)

model_path = "tensorflow_data/CNN_model.h5"
model.save(model_path)


##############################################################################
# 숫자분류 개선 작업을 위한 데이터 처리 연습


import tensorflow as tf
from tensorflow import keras
import os
from . my_modules import *
import numpy as np
import cv2
import copy

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# localized_train_images = copy.deepcopy(train_images)
# localized_test_images = copy.deepcopy(test_images)

def localize_image(image_data, empty_value=0, size=28):
    # 이미지 데이터셋은 3차원 데이터이어야 한다.
    localized_image_data = copy.deepcopy(image_data)
    data_n = localized_image_data.shape[0]
    for i in range(data_n):
        img = localized_image_data[i]
        # 박스의 바운더리 좌표 계산
        b_result = get_boundary(img)
        new_height = b_result[2] - b_result[0] + 1
        new_width = b_result[3] - b_result[1] + 1
        # if new_height > new_width:
        #     max_length = new_height
        # else:
        #     max_length = new_width
        # 검색된 영역대로 잘라낸 데이터를 저장.
        img = img[b_result[0]:(b_result[2] + 1), b_result[1]:(b_result[3] + 1)]
        img = 255 - img
        img = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_AREA)
        localized_image_data[i] = 255 - img
        if i % 2000 == 0:
            print(i)
    return localized_image_data

new_train_images = localize_image(train_images)
new_test_images = localize_image(test_images)

new_train_images = new_train_images.reshape((60000, 28, 28, 1))
new_test_images = new_test_images.reshape((10000, 28, 28, 1))

new_train_images = new_train_images / 255.0
new_test_images = new_test_images / 255.0

model.fit(new_train_images, train_labels, epochs=5)
model.evaluate(new_test_images, test_labels, verbose=2)

model_path = "tensorflow_data/NR_basic_CNN_adjusted.h5"
model.save(model_path)

print_number(new_train_images[0], 28, 28, 2)





src = cv2.imread("static/images/GANN_life.jpg", cv2.IMREAD_COLOR)
src = cv2.imread("static/images/GANN_life.jpg", cv2.IMREAD_GRAYSCALE)
dst = cv2.resize(src, dsize=(28, 28), interpolation=cv2.INTER_AREA)

dst2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow("src", src)
# cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()