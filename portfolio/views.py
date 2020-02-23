from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files import File

from .models import TensorflowModel, UploadImageModel

from .my_modules import *
from .forms import UploadFileForm

import numpy as np
from PIL import Image

import subprocess
import os


# 텐서플로 모델을 저장하는 전역변수.
TENSORFLOW_MODELS = dict()

def main_page(request):
    return render(request, "portfolio/index.html")

def number_recognition_main_page(request):
    # POST 된 데이터 확인.
    requested_data_dict = dict(request.POST)
    requested_data_dict_key_list = list(requested_data_dict.keys())

    # 현재 로드되어 있는 모델이 없는 경우.
    # 최초에만 실행된다.
    if len(TENSORFLOW_MODELS) == 0:
        # html 에 context 전달을 하기 위해 쿼리를 준다.
        tf_load_model_info = TensorflowModel.objects.all()

        # 저장된 모델을 불러온다.
        for model in tf_load_model_info:
            TENSORFLOW_MODELS[model.model_type] = dict()
            TENSORFLOW_MODELS[model.model_type]["model_name"] = model.model_name
            TENSORFLOW_MODELS[model.model_type]["model_type"] = model.model_type
            TENSORFLOW_MODELS[model.model_type]["model"] = load_tensorflow_keras_model(model.file_path)
            TENSORFLOW_MODELS[model.model_type]["description"] = model.description
            TENSORFLOW_MODELS[model.model_type]["prediction_result"] = ""

    # POST 데이터가 있는 경우.
    if len(requested_data_dict_key_list) > 0:

        if "pixel_data" in requested_data_dict_key_list:
            pixel_data = requested_data_dict.get("pixel_data")
            pixel_data = np.array(pixel_data, dtype=np.float32)
            pixel_data = pixel_data / 255

            # 콘솔에 숫자가 보이도록 출력.
            # print_number(pixel_data)

            # 모델명 전체 저장.
            model_names = list(TENSORFLOW_MODELS.keys())
            for i in range(len(model_names)):
                cur_model_type = model_names[i]
                cur_model = TENSORFLOW_MODELS[cur_model_type]["model"]
                prediction_data = None

                # 어떤 모델이냐에 따라 input의 형태를 다르게 해야함.
                if cur_model_type in ["NR_basic"]:
                    prediction_data = pixel_data.reshape(-1, 28 * 28)
                elif cur_model_type in ["NR_basic_CNN"]:
                    prediction_data = pixel_data.reshape(-1, 28, 28, 1)
                elif cur_model_type in ["NR_basic_CNN_adjusted"]:
                    prediction_data = pixel_data.reshape(-1, 28, 28)
                    prediction_data = localize_image(prediction_data, max_value=1.0)
                    prediction_data = prediction_data.reshape(-1, 28, 28, 1)

                if prediction_data is not None:
                    result = cur_model.predict(prediction_data)
                    TENSORFLOW_MODELS[cur_model_type]["prediction_result"] = str(np.argmax(result))

                # print("모델 " + str(i) + ". 분석결과 : " + str(np.argmax(result)))





    # html render 시 전달하는 context는 iterable한 변수이어야 한다.
    # context의 key값이 template에서 사용하는 변수의 이름이 된다.
    # 딕셔너리와 리스트를 전달할 때의 차이점에 주의.
    # 리스트를 전달해야 정상적으로 읽을 수 있음.(혹은 아래와 같이 딕셔너리에서 value들만 전달하는 방법도 가능)
    context = {"NR_models": TENSORFLOW_MODELS.values()}
    # print(context)

    return render(request, "portfolio/number_recognition.html", context)


def GANN_main_page(request):
    context = {}
    return render(request, "portfolio/GANN.html", context)

def object_detection_main_page(request):
    # file input을 통해 전달되는 post 데이터는 request.FILES 에서 볼 수 있다.

    # 렌더링 하는 context변수
    context = {}

    # ModelForm 생성
    form = UploadFileForm()

    # 분석하려는 이미지를 업로드 한 경우
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # save()를 실행하면, return으로 저장한 데이터 객체를 받는다.
            saved_image = form.save()

            # 용량이 큰 파일이 HEROKU에 업로드 되지 않아 해결책으로 분할하여 업로드 하는 방법을 사용함.
            # weight 파일이 만들어져 있지 않은 경우, 분할된 파일을 병합하여 만든다.
            weight_filename = "yolov2.weights"
            file_list = os.listdir(os.getcwd())

            if weight_filename not in file_list:
                with open(weight_filename, 'wb') as f_write:
                    split_file_name = ['yolov2_01.weights', 'yolov2_02.weights', 'yolov2_03.weights']
                    for i in range(3):
                        with open(split_file_name[i], 'rb') as f_read:
                            lines = f_read.readlines()
                        f_write.writelines(lines)

            # darknet을 cmd 창에서 직접 실행한다.
            # 개발환경에서는 windows cmd.
            # heroku 배포환경에서는 UNIX.
            cmd = "darknet_no_gpu detector test data/coco.data yolo.cfg yolov2.weights "
            # cmd = "./darknet_no_gpu detector test data/coco.data yolo.cfg yolov2.weights "
            cmd += saved_image.image.url[1:]

            # 화면에 출력되는 모든 결과를 텍스트로 저장.
            result = subprocess.check_output(cmd)

            # 생성된 분석 파일을 저장하기 위해 파일명 추출.
            result_str = str(result)
            result_str_pos = result_str.index("predictions_")
            prediction_image_filename = result_str[result_str_pos:result_str_pos + 32] + ".jpg"


            image_file_instance = UploadImageModel()
            f = File(open(prediction_image_filename, "rb"))
            image_file_instance.image.save(prediction_image_filename, f)

            prediction_image_form = UploadFileForm(request.POST, instance=image_file_instance)
            prediction_image = prediction_image_form.save()


            context["form"] = form
            context["saved_image"] = saved_image
            context["prediction_image"] = prediction_image

            # os.remove(prediction_image_filename)

            return render(request, "portfolio/object_detection.html", context)

    # 이미지를 업로드 하지 않은 경우에는 그냥 그대로 보여준다.
    context["form"] = form

    return render(request, "portfolio/object_detection.html", context)