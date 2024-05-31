from .back_end import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  


@app.route('/')
def home():
  return render_template('index.html')


import time

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global vidPath,summarization
    try:
        vidPath=saveFile(app,request)
        return {"Message": 'Video uploaded Done and Procces start....!'}
    except Exception as e:
        return {"Error": e}
    finally:
        apply_model()
        delete_dir_contents("uploads")
        return{"Summerizition":summarization}
    
@app.route('/chat/<msg>', methods=['POST','GET'])
def chat(msg):
    if not vidPath or not visual_output or not audio_output:
        return Gemini.generate_content(msg).text
    return Gemini.generate_content(f'if this is the visual description of the video "{visual_output}" , and this is the audio part "{audio_output}" \n اجب على هذا السؤال،السؤال: {msg} اذا كان').text



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
