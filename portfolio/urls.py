from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from django.views.static import serve
from django.conf.urls import url
from . import views


urlpatterns = [
	# 포트폴리오 메인 페이지
    path("", views.main_page, name="main_page"),

    # 숫자인식 메인 페이지
    path("number_recognition/", views.number_recognition_main_page, name="number_recognition_main_page"),

    # GANN(유전알고리즘+인공신경망) 메인 페이지
    path("GANN/", views.GANN_main_page, name="GANN_main_page"),

    # 물체 인식 메인 페이지
    path("object_detection/", views.object_detection_main_page, name="object_detection_main_page"),

    # 머신러닝 연습 페이지
    path("ml_practice/", views.ml_practice_main_page, name="ml_practice_main_page"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# urlpatterns += url(r'^media/(?P<path>.\*)$', serve, {
#     'document_root': settings.MEDIA_ROOT,
# })