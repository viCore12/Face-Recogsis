import pandas as pd
import time
import base64
import torch
import timm
import tempfile
import os
import av
import cv2

from model import FaceModel
from config import opt
from predict_yolo import predict_yolo, FaceData
from predict import predict, predict_result
from visualize import visualize_image

import streamlit as st
from torchvision import transforms
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Face Detection Demo", page_icon="👤")

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

yolo_model = YOLO(opt['yolo_path'])

vit_model = timm.create_model('vit_base_patch16_384', pretrained=False)
vit_model.head = torch.nn.Linear(768,2016,bias = True)
vit_model.to(opt['device'])
vit_model.load_state_dict(torch.load(opt['model_vit_path'], map_location=torch.device(opt['device'])))

model = FaceModel(opt['num_age'], opt['num_race'], opt['num_masked'], opt['num_skintone'], opt['num_emotion'], opt['num_gender'], 'test').to(opt['device'])

model.load_state_dict(torch.load(opt['model_path'], map_location=torch.device(opt['device'])))

class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)
        for x, y, w, h in faces:
            bbox = x, y, w, h
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((384,384),antialias=True),
                                    ])
            data = FaceData(frm, bbox, os.getcwd(), transform, stage='Video')
            pre = predict(data, model, opt['device'])
            result = predict_result(pre)
            age = result['age'].item()
            race = result['race'].item()
            masked = result['masked'].item()
            skintone = result['skintone'].item()
            emotion = result['emotion'].item()
            gender = result['gender'].item()

            info_str = f"Gender: {gender}, Age: {age}, Race: {race}, Emotion: {emotion}, Skintone: {skintone}, Mask: {masked}"
            print(info_str)

            gender_str = f"Gender: {gender}"
            age_str = f"Age: {age}"
            race_str = f"Race: {race}"
            emotion_str = f"Emotion: {emotion}"
            skintone_str = f"Skintone: {skintone}"
            masked_str = f"Masked: {masked}"

            # Đặt nhãn cho mỗi thông tin
            labels = [gender_str, age_str, race_str, emotion_str, skintone_str, masked_str]

            # Xác định vị trí của nhãn
            label_x = x - 150
            label_y = y - 20  # Đặt nhãn bên trái khung khuôn mặt

            # Vẽ nhãn cho mỗi thông tin
            for label in labels:
                cv2.putText(frm, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                label_y += 20  # Xuống dòng

            cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 3)

            #Tính FPS
            # self.frame_count += 1
            # current_time = time.time()

            # if current_time - self.start_time >= 1:
            #     fps = self.frame_count / (current_time - self.start_time)
            #     print("FPS:", fps)
            #     self.frame_count = 0
            #     self.start_time = current_time

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

# thêm lựa chọn upload ảnh hoặc camera
st.sidebar.title("Face Detection")

option = st.sidebar.selectbox('Select option',('Upload image', 'Camera', 'Video'))

if option == 'Upload image':
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg','png','jpeg'])
elif option == 'Camera':
    uploaded_file = st.camera_input(label="Camera")
elif option == 'Video':
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    uploaded_file = True
    webrtc_streamer(key="key", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIGURATION)


if uploaded_file is not None:
    if option == 'Upload image' or option == 'Camera':
        if option == 'Upload image':
            image_path = os.path.join(os.getcwd(), uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif option == 'Camera':
            image_path = os.path.join(os.getcwd(), 'image.jpg')
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        t1 = time.time()
        results = yolo_model((image_path), stream=True, device = opt['device'], verbose=False)
        filenames, bboxs = predict_yolo(results)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((384,384),antialias=True),
                                    ])

        st.header("Ảnh gốc")

        st.image(uploaded_file, use_column_width=True)

        for filename, bbox in zip(filenames, bboxs):
            data = FaceData(filename, bbox, os.getcwd(), transform, stage='Image')
            pre = predict(data, model, opt['device'])
            result = predict_result(pre)

            t2 = time.time()

            t = round(t2-t1, 2)

            st.write(f'Inference time: {t} seconds') 
                
            visualize_image(result)

    # elif option == 'Video':

st.sidebar.header('About')
st.sidebar.info('This is a demo for face recognition')
st.sidebar.header('How to use')
st.sidebar.markdown("""
1. Upload your image.
2. Wait for the result.
""")
st.sidebar.header('Hardware')
st.sidebar.markdown("""
- CPU: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz   2.69 GHz
- RAM: 16.0 GB
""")

# set background image
main_bg = "background.png"
main_bg_ext = "png"

side_bg = "background.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"]
    {{
        background-image: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        background-size: cover;
    }}
    [data-testid="stFullScreenFrame"] div
    {{
        display: flex;
        justify-content: center !important;
        align-items: center;
    }}
    [data-testid="stFullScreenFrame"] [data-testid="stImage"] img
    {{
        object-fit: contain !important;
        width: 80% !important;
        height: 80% !important;
    }}
    [data-testid="block-container"]
    {{
        max-width: 100% !important;
        max-height: 100% !important;
    }}
    [data-testid="stBlock"]
    {{
        width: 100% !important;
        height: 100% !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)