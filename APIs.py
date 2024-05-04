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
import serverFunctions as SF

audio_text,visual_output="",""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  
 

@app.route('/AIchat/<string:q>', methods=['GET'])
def chatAI(q):
    global audio_text,visual_output
    response = SF.AI_VideoChat(visual_output,audio_text,q)
    return {"message":response}


@app.route('/upload_video', methods=['POST'])
def upload_video():
    global audio_text,visual_output
    file_path=SF.saveFile(app,request)
    audio_text,visual_output=SF.all_process(file_path)
    response = SF.AI_VideoChat(visual_output, audio_text, "لخص لي الفيديو ده")
    return {"message":"ملخص الفيديو:\n"+response}



if __name__ == '__main__':
    app.run(debug=True)
