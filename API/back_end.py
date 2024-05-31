import os
import cv2
import math
from PIL import Image
from threading import Thread
import speech_recognition as sr
from scipy.signal import find_peaks
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from moviepy.video.io.VideoFileClip import VideoFileClip
from flask import Flask, request, jsonify,render_template
from transformers import BlipProcessor, BlipForConditionalGeneration
API = os.environ.get('G_API_KEY')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
genai.configure(api_key=API)
Gemini = genai.GenerativeModel(model_name= 'gemini-pro')
vidPath,audio_output,visual_output,summarization="","","",""




def delete_dir_contents(directory):

  if not os.path.isdir(directory):
    return
  for root, dirs, files in os.walk(directory, topdown=False):
    for file in files:
      try:
        os.remove(os.path.join(root, file))
      except :
        pass
    for dir in dirs:
      try:
        os.rmdir(os.path.join(root, dir))
      except :
        pass


def transcribe():
        global audio_output
        video = VideoFileClip("uploads/target.mp4")
        audio = video.audio
        audio_file = os.path.join( "extracted.wav")
        audio.write_audiofile(audio_file)
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='ar')
            audio_output=text
            return 
        except sr.UnknownValueError:
            return "Google Web Speech API could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Web Speech API; {e}"
        finally:
            os.remove(audio_file)



def trim_and_extract_frames(output_path, start_time=0, end_time=None, frame_interval=2):
    """Trims a video clip, writes it to an output file, and extracts frames at specified intervals."""
    video_clip = VideoFileClip("uploads/target.mp4")  # Open the video
    if end_time is None:
        end_time = start_time + 300  # Set end time if not provided (default: 300 seconds)
    trimmed_clip = video_clip.subclip(start_time, end_time)  # Trim the clip
    trimmed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")  # Write the trimmed clip
    
    # Frame extraction
    os.makedirs('Frames', exist_ok=True)  # Create output directory if it doesn't exist
    cap = cv2.VideoCapture(output_path)
    
    # Get video length in seconds using FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Could not determine FPS, estimation might be inaccurate.")
    video_length_seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    # Calculate target number of frames based on video length and interval
    num_frames = int(math.ceil(video_length_seconds / frame_interval))

    selected_frames = []
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % num_frames == 0:
            selected_frames.append(frame.copy())
        current_frame += 1

    segment_id = os.path.splitext(os.path.basename(output_path))[0].split('_')[-1]
    for i, frame in enumerate(selected_frames):
        frame_filename = os.path.join("Frames", f"frame{segment_id}_{i}.jpg")
        cv2.imwrite(frame_filename, frame)
    
    cap.release()
    video_clip.close()  # Close the original video clip
    trimmed_clip.close()  # Close the trimmed clip

def split_video(segment_duration=180, frame_interval=6):
    """Splits a video into segments and extracts frames using threads for concurrent processing."""
    video_clip = VideoFileClip("uploads/target.mp4")  # Open the video
    total_duration = int(video_clip.duration)  # Get total duration
    video_clip.close()  # Close the original video clip

    num_segments = total_duration // segment_duration
    if total_duration % segment_duration != 0:
        num_segments += 1  # Handle remaining segment if duration not divisible by segment length

    threads = []  # List to store threads
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)
        output_path = f"SubVideos/vid_segment_{i+1}.mp4"
        thread = Thread(target=trim_and_extract_frames, args=(output_path, start_time, end_time, frame_interval))  # Create thread
        threads.append(thread)
        thread.start()  # Start the thread immediately

    # Wait for all threads to finish
    for thread in threads:
        thread.join()  

# Example usage





def generate_captions():
        global visual_output
        text = "all details in this image are : "
        all_unconditional = ''
        for filename in os.listdir("Frames"):
            if filename.endswith(".jpg"):
                image_path = os.path.join("Frames", filename)
                raw_image = Image.open(image_path).convert('RGB')
                inputs = processor(raw_image, text, return_tensors="pt")
                out = model.generate(**inputs)
                inputs = processor(raw_image, return_tensors="pt")
                out = model.generate(**inputs)
                unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

                visual_output = all_unconditional + ". " + unconditional_caption

        return



def saveFile(app,request):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file :
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename("target.mp4")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path


def apply_model():
    global visual_output,audio_output,summarization
    # Define threads for each function
    transcribe_thread = Thread(target=transcribe)
    split_video_thread = Thread(target=split_video)
    generate_captions_thread = Thread(target=generate_captions)

    # Start threads
    transcribe_thread.start()
    split_video_thread.start()

    # Wait for split_video_thread to finish
    split_video_thread.join()

    # Start generate_captions_thread after split_video_thread finishes
    generate_captions_thread.start()

    # Wait for transcribe_thread and generate_captions_thread to finish
    transcribe_thread.join()
    generate_captions_thread.join()

    # Call the reset function after all threads are done
    delete_dir_contents("Frames")
    delete_dir_contents("SubVideos")
    visual_output = Gemini.generate_content(f'the following is description of the key frames of video separated by "." , combine them into one description in arabic without repititions \n{visual_output}')
    visual_output = visual_output.text
    response = Gemini.generate_content(f'if this is the visual description of the video {visual_output} , and this is the audio part {audio_output} \n لخص لي الفيديو ده')
    summarization=response.text
    return("ملخص الفيديو:\n"+summarization)


