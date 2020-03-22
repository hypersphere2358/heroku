from django.contrib import admin
from .models import TensorflowModel
from .models import MachineLearningTable

# Register your models here.

admin.site.register(TensorflowModel)
admin.site.register(MachineLearningTable)