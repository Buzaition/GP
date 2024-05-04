#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
from transformers import BlipProcessor, BlipForConditionalGeneration
from moviepy.video.io.VideoFileClip import VideoFileClip
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
import google.generativeai as genai
from scipy.signal import find_peaks
import speech_recognition as sr
from PIL import Image
import numpy as np
import torch
import cv2
import os
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑



#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
API_KEY = os.environ.get('API_KEY')
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name= 'gemini-pro')
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

def saveFile(app,request):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file :
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path



def AI_VideoChat(visual_output:str,audio_output:str,question:str):
    response=model.generate_content(f'if this is the visual description of the video {visual_output} , and this is the audio part {audio_output} \n اجب على هذا السؤال،السؤال: {question}')
    return response.text


def AI_Chat(question:str):
    response=model.generate_content(question)
    return response.text



def trim_video(input_path, output_path, start_time=0, end_time=None):
        video_clip = VideoFileClip(input_path)
        if end_time is None:
            end_time = start_time + 300
        trimmed_clip = video_clip.subclip(start_time, end_time)
        trimmed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        video_clip.close()
        trimmed_clip.close()




def extract_audio(video_path, output_dir):
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_filename = os.path.join(output_dir, "extracted.wav")
        audio.write_audiofile(audio_filename)
        return audio_filename



def transcribe_audio(audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='ar')
            return text
        except sr.UnknownValueError:
            return "Google Web Speech API could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Web Speech API; {e}"



def representative_frame_sampling(video_path, output_dir, num_frames=10):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_interval = max(frame_count // num_frames, 1)
        selected_frames = []
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % skip_interval == 0:
                selected_frames.append(frame.copy())
            current_frame += 1
        for i, frame in enumerate(selected_frames):
            frame_filename = os.path.join(output_dir, f"representativeframe{i}.jpg")
            cv2.imwrite(frame_filename, frame)
        cap.release()




def compute_frame_difference(prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        score = np.sum(frame_diff)
        return score





def select_frames(video_path, output_dir, max_frames=10, frame_skip=10):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_scores = []
        prev_frame = None
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                frames.append(frame)
                if prev_frame is not None:
                    score = compute_frame_difference(prev_frame, frame)
                    frame_scores.append(score)
                prev_frame = frame
            frame_count += 1
        cap.release()
        frame_scores = np.array(frame_scores)
        peak_indices, _ = find_peaks(frame_scores, distance=frame_skip)
        if len(peak_indices) > max_frames:
            sorted_indices = np.argsort(frame_scores)[::-1]
            peak_indices = sorted_indices[:max_frames]
        selected_frames = [frames[i] for i in peak_indices]
        for i, frame in enumerate(selected_frames):
            frame_filename = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_filename, frame)
        return selected_frames




def generate_captions(folder_path):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        text = "all details in this image are : "
        all_conditional = ''
        all_unconditional = ''
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(folder_path, filename)
                raw_image = Image.open(image_path).convert('RGB')
                inputs = processor(raw_image, text, return_tensors="pt")
                out = model.generate(**inputs)
                conditional_caption = processor.decode(out[0], skip_special_tokens=True)
                inputs = processor(raw_image, return_tensors="pt")
                out = model.generate(**inputs)
                unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
                all_conditional = all_conditional + ". " + conditional_caption
                all_unconditional = all_unconditional + ". " + unconditional_caption
        return all_conditional, all_unconditional




def all_process(input_vid_path):
    # Start the processing
    output_dir = os.path.dirname(input_vid_path)
    audio_file = extract_audio(input_vid_path, output_dir)
    audio_text = transcribe_audio(audio_file)
    video_output_path = os.path.splitext(input_vid_path)[0] + "_trimmed.mp4"
    trim_video(input_vid_path, video_output_path, start_time=0, end_time=60)
    frame_output_dir = os.path.join(output_dir, "frames")
    representative_frame_sampling(video_output_path, frame_output_dir, num_frames=10)
    conditional_text, unconditional_text = generate_captions(frame_output_dir)
    response = model.generate_content(f'the following is description of the key frames of video separated by "." , combine them into one description in arabic without repititions \n{unconditional_text}')
    visual_output = response.text
    return audio_text,visual_output

