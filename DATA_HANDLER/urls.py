from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path("addSensorData/", addSensorData),
    path("addImageData/", addImageData),
    path('playMusic/', play_audio),
    path('stressDetector', detect_stress),
    path('predict/', predict)
]
