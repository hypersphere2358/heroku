from django.db import models

# Create your models here.

class TensorflowModel(models.Model):
    # 텐서플로 모델의 이름.
    model_name = models.CharField(max_length=30)

    # 텐서플로 모델을 저장하고 있는 파일명.
    # "tensorflow_data/"가 최초 경로이다.
    file_path = models.CharField(max_length=30)

    # 모델이 로드되었는지 상태를 나타내는 필드.
    load_status = models.BooleanField(default=False)

    # # 모델 파일이 존재하는지 여부
    # file_eixst = models.BooleanField()

    # 모델 설명.
    description = models.TextField()

    def __str__(self):
        return self.model_name


class UploadImageModel(models.Model):
    name = models.CharField(max_length=255, default="temp_image")
    image = models.ImageField(upload_to="")