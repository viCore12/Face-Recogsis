import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO
from itertools import cycle

def visualize_image(sample):
    image = sample['image'].item().numpy().transpose(1, 2, 0)

    age = sample['age'].item()
    race = sample['race'].item()
    masked = sample['masked'].item()
    skintone = sample['skintone'].item()
    emotion = sample['emotion'].item()
    gender = sample['gender'].item()

    labels = {
        'Age': age,
        'Race':race,
        'Masked': masked,
        'Skintone': skintone,
        'Emotion': emotion,
        'Gender': gender
    }

    st.header("Kết quả")

    col1, col2 = st.columns(2)

    # Display image in the first column with some styling
    col1.image(image, use_column_width=True)

    # Display labels in the second column with styling
    for key, value in labels.items():
        col2.text(f'{key}: {value}')

    st.markdown(
        """
        <style>
        [data-testid="stHorizontalBlock"]{
            max-width: 100% !important;
            max-height: 100% !important;
        }
        [data-testid="stImage"] img
        {
            max-width: 90% !important;
            max-height: 90% !important;
        }
        [data-testid="stText"]{
            font-size: 20px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

        
