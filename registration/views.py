
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        file_name = default_storage.save('videos/' + video_file.name, ContentFile(video_file.read()))
        file_url = default_storage.url(file_name)
        print("this is the file stored in the backend media",file_url)
        return JsonResponse({'file_url': file_url}, status=200)
    return JsonResponse({'error': 'Invalid request'}, status=400) 