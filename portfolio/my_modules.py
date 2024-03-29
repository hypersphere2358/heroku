import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
# import cv2
import copy
from PIL import Image
from myportfolio import settings


TENSORFLOW_DATA_DIR = "tensorflow_data/"
TEMPORARY_IMAGE_DIR = "temp_img_data/"

def load_tensorflow_keras_model(model_file_path):

    try:
        model_path = TENSORFLOW_DATA_DIR + model_file_path
        model = keras.models.load_model(model_path)
    except:
        model = None

    return model


# 28*28 개의 배열을 받으면 숫자로 출력
def print_number(x, row_n, col_n, type=1):
    x = np.array(x)
    # 데이터 체크.
    if type == 1 and len(x) != row_n * col_n:
        print("Array length error.")
        return
    if type == 2:
        row_n = x.shape[0]
        col_n = x.shape[1]
    for i in range(row_n):
        s = ""
        for j in range(col_n):
            if type == 1:
                if x[i * col_n + j] == 0:
                    s += "0"
                else:
                    s += "1"
            elif type == 2:
                if x[i][j] == 0:
                    s += "0"
                else:
                    s += "1"
        print(s)


def get_boundary(array_data):
    array_data = np.array(array_data)
    row_n = array_data.shape[0]
    col_n = array_data.shape[1]
    col_min = 999999
    col_max = -999999
    row_min = 999999
    row_max = -999999
    for i in range(row_n):
        for j in range(col_n):
            if array_data[i][j] > 0:
                if i < row_min:
                    row_min = i
                if i > row_max:
                    row_max = i
                if j < col_min:
                    col_min = j
                if j > col_max:
                    col_max = j
    boundary_result = list()
    boundary_result.append(row_min)
    boundary_result.append(col_min)
    boundary_result.append(row_max)
    boundary_result.append(col_max)
    return boundary_result

def localize_image(image_data, empty_value=0, size=28, max_value=255.0):
    # 이미지 데이터셋은 3차원 데이터이어야 한다.
    localized_image_data = copy.deepcopy(image_data)
    data_n = localized_image_data.shape[0]
    for i in range(data_n):
        img = localized_image_data[i]

        # 참고. pillow 데이터에서 흑백 이미지 데이터 : 0=검정, 255=흰색
        # 하지만 view.py에서 255로 나눈 후 함수로 들어오기 때문에 0~1이 된다.

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

        # pillow로 resize처리
        img = max_value - img
        img *= 255
        pillow_image = Image.fromarray(img)
        pillow_image = pillow_image.resize((size, size))
        img = np.array(pillow_image)
        img /= 255
        img = max_value - img

        # opencv로 resize 처리
        # heroku에서 import cv2를 처리하지 못하므로 사용하지 않음.
        # img = max_value - img
        # img = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_AREA)
        localized_image_data[i] = img
        # if i % 2000 == 0:
        #     print(i)
    return localized_image_data

def temp_image_save(img_data, filename):
    img = Image.open(img_data)
    # 이미지가 너무 큰 경우(높이 400초과) 높이를 400으로 맞춘다.
    # 즉 이미지의 최대 높이를 400으로 제한한다.
    if img.height > 250:
        new_h = 250
        new_w = int(new_h / img.height * img.width)
        img = img.resize((new_w, new_h))

    file_path = os.path.join(settings.STATIC_ROOT, TEMPORARY_IMAGE_DIR, filename)
    img.save(file_path)
    # file_path = os.path.join(TEMPORARY_IMAGE_DIR, filename)

    return file_path

# 2020.03.19 부터 작성중. c로 된 darknet을 python으로 옮기기
def yolo_python():

    return True