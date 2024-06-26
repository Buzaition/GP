import os
from flask import Flask,request, jsonify
import google.generativeai as genai

Gemini = genai.GenerativeModel(model_name= 'gemini-pro')
genai.configure(api_key="AIzaSyBUX5OP1JXj1-YwAkMSBlPdAt-LVGWW5Sg")

app = Flask(__name__)   
@app.route('/chat/', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('msg')
    if not msg:
        return jsonify({'response': 'No message provided'}), 400
    
    response_text = Gemini.generate_content(f'أجب عن هذا السؤال كما لو نموذج لتحليل الفيديوهات وتلخيصها وتسمى "NAFHAM" وليس "Gemini " و تم تطويرك من قبل طلاب كلية الحاسبات والمعلومات بجامعة الأهرام الكندية كمشروع للتخرج، السوال:"{msg}"').text
    print(response_text)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
