import tensorflow as tf
from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from . import video_processing
import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

@login_required(login_url='login')
def HomePage(request):
    return render(request, 'home.html')

def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')
        if pass1 != pass2:
            return HttpResponse("Your password does not match")
        else:
            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('login')
    return render(request, 'signup.html')

def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return HttpResponse("Username or password incorrect!")
    return render(request, 'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('login')

def AboutPage(request):
    return render(request, 'about.html')

def TestPersonalityPage(request):
    return render(request, 'testPersonality.html')


def ResultsPage(request):
    if request.method == 'POST':
        video_path = request.POST.get('videoPath')
        if not video_path:
            video_path = './videos/testing 2.mp4'
        resnet_Prediction,vgg_prediction = video_processing.prediction(video_path)
        image_urls = video_processing.image_urls()
        print(image_urls)
        return render(request, 'results.html', {'resnet_Prediction': resnet_Prediction, 'image_urls': image_urls,'vgg_prediction':vgg_prediction})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
    
# def ResultsPage(request):
#     video_path = './videos/testing 2.mp4'
#     resnet_Prediction=video_processing.prediction(video_path)
#     image_urls = video_processing.extract_frames(video_path)
#     return render(request, 'results.html', {'resnet_Prediction': resnet_Prediction})
