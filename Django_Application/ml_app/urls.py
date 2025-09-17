from django.urls import path
from . import views

app_name = 'ml_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('video-upload/', views.video_upload_page, name='video_upload'),
    path('audio-upload/', views.audio_upload_view, name='audio_upload'),
    path('upload-image/', views.image_upload_view, name='upload_image'),
    path('image-result/', views.image_upload_view, name='image_result'),
    path('about/', views.about, name='about'),
    path('predict/', views.predict_page, name='predict'),
    path('cuda_full/', views.cuda_full, name='cuda_full'),
    path('prediction-details/', views.prediction_details_view, name='prediction_details'),
]
