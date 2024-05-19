import librosa
import numpy as np
import cv2
import random
import tensorflow as tf
from typing import Tuple
from moviepy.editor import VideoFileClip
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import os
import subprocess
from django.conf import settings

BASE_DIR = settings.BASE_DIR
MEDIA_ROOT = settings.MEDIA_ROOT
FFMPEG_PATH = os.path.join(BASE_DIR, 'ffmpeg', 'bin', 'ffmpeg.exe')

print("This is the base directory: ", BASE_DIR)
print("FFMPEG path: ", FFMPEG_PATH)
print("Media root: ", MEDIA_ROOT)

# Custom layer definition
@register_keras_serializable(package='Custom', name='FrameExtractionLayer')
class FrameExtractionLayer(Layer):
    def __init__(self, index, **kwargs):
        super(FrameExtractionLayer, self).__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index]

    def get_config(self):
        config = super(FrameExtractionLayer, self).get_config()
        config.update({"index": self.index})
        return config

# Load model with custom objects
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, 'models/resnet_model_v1.5.keras'),
    custom_objects={'FrameExtractionLayer': FrameExtractionLayer}
)

def convert_webm_to_mp4(webm_path):
    mp4_path = webm_path.replace('.webm', '.mp4')
    subprocess.run([FFMPEG_PATH, '-i', webm_path, mp4_path])
    return mp4_path

def extract_audio_from_video(video_path):
    audio_dir = os.path.join(BASE_DIR, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, os.path.basename(video_path).replace('.mp4', '.wav'))

    with VideoFileClip(video_path) as video:
        audio = video.audio
        audio.write_audiofile(audio_path, codec='pcm_s16le')
    
    return audio_path

def preprocess_audio_series(raw_data: np.ndarray) -> np.ndarray:
    N, M = 24, 1319
    mfcc_data = librosa.feature.mfcc(y=raw_data, n_mfcc=24)
    mfcc_data_standardized = (mfcc_data - np.mean(mfcc_data)) / np.std(mfcc_data)
    
    if mfcc_data_standardized.shape[1] > M:
        mfcc_data_standardized = mfcc_data_standardized[:, :M]  # Truncate
    else:
        number_of_columns_to_fill = M - mfcc_data_standardized.shape[1]
        padding = np.zeros((N, number_of_columns_to_fill))
        mfcc_data_standardized = np.hstack((mfcc_data_standardized, padding))  # Pad
    
    return mfcc_data_standardized.reshape(N, M, 1)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    static_dir = os.path.join(BASE_DIR, 'templates', 'static', 'images')
    print("\n\n\n", static_dir)
    os.makedirs(static_dir, exist_ok=True)
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    interval = 1.5 * fps  # Interval of 2 seconds in terms of frame count

    for i in range(6):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)  # Set position to the i-th interval
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(static_dir, f"frame_{i}.jpg")
            cv2.imwrite(img_path, frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def image_urls():
    static_dir = os.path.join(BASE_DIR, 'templates', 'static', 'images')
    urls = []
    for file_name in os.listdir(static_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            url_path = os.path.join('/static/images', file_name)
            urls.append(url_path)
    
    return urls

def resize_image(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def crop_image_window(image: np.ndarray, training: bool = True) -> np.ndarray:
    height, width, _ = image.shape
    if training:
        MAX_N = height - 128
        MAX_M = width - 128
        rand_N_index, rand_M_index = random.randint(0, MAX_N), random.randint(0, MAX_M)
        return image[rand_N_index:(rand_N_index+128), rand_M_index:(rand_M_index+128), :]
    else:
        N_index = (height - 128) // 2
        M_index = (width - 128) // 2
        return image[N_index:(N_index+128), M_index:(M_index+128), :]

def prediction(path):
    full_path = os.path.join(MEDIA_ROOT, path.lstrip('/').lstrip('media/'))
    print("This is the full path: ", full_path)
    
    if full_path.endswith('.webm'):
        full_path = convert_webm_to_mp4(full_path)
    
    audio_path = extract_audio_from_video(full_path)
    audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    preprocessed_audio = preprocess_audio_series(raw_data=audio_data)
    frames = extract_frames(full_path)
    resized_images = [resize_image(image=im, new_size=(128, 128)) for im in frames]
    cropped_images = [crop_image_window(image=resi, training=True) / 255.0 for resi in resized_images]
    preprocessed_video = np.stack(cropped_images)
    predicted_personality_traits = model.predict([preprocessed_audio.reshape(1, 24, 1319, 1), preprocessed_video.reshape(1, 6, 128, 128, 3)])
    personalities = ['Neuroticism', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'Openness']
    for label, value in zip(personalities, predicted_personality_traits[0]):
        print(label + ': ' + str(value))
    predicted_traits = {label: value for label, value in zip(personalities, predicted_personality_traits[0])}
    
    return predicted_traits

# Example usage
# prediction('videos/recorded_video_aSGTKdC.webm')
