"""
URL configuration for registration project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from http import server
from django.conf import settings
from django.contrib import admin
from django.urls import path
from loginapp import views
from .views import * 
from django.contrib.staticfiles.views import serve
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.SignupPage, name='signup'),
    path('Login/', views.LoginPage, name='login'),
    path('home/', views.HomePage, name='home'),
    path('logout/', views.LogoutPage, name='logout')  ,
    path('About/', views.AboutPage, name='about'),
    path('Test-Personality/', views.TestPersonalityPage, name='testPersonality'),
    # path('extracted-frames/', views.extracted_frames_view, name='extracted_frames'),
    path('Test-Results/', views.ResultsPage, name='testResults'),
    # path('Test-Personality/', views.save_video, name='videoDownload'),
    path('static/', serve, {'document_root': settings.STATIC_ROOT}),
    path('upload-video/', upload_video, name='upload_video'),
     
]
