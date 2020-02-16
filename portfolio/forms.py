from django import forms
from .models import UploadImageModel

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadImageModel
        fields = ('image', )