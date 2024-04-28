import os
import torch
import pandas as pd

df = pd.read_csv('data/labels.csv')

opt = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_path': 'data/train.csv',
    'test_path': 'data/test.csv',
    'valid_path': 'data/valid.csv',
    'image_train_path': 'data/data_yolo/data_yolo/train/images',
    'image_test_path': 'data/data_yolo/data_yolo/test/images',
    'image_valid_path': 'data/data_yolo/data_yolo/valid/images',
    'num_age': len(df['age'].value_counts()),
    'num_gender': len(df['gender'].value_counts()),
    'num_race': len(df['race'].value_counts()),
    'num_masked': len(df['masked'].value_counts()),
    'num_skintone': len(df['skintone'].value_counts()),
    'num_emotion': len(df['emotion'].value_counts()),
    'encoding_mapping': {'age': {'20-30s': 0, '40-50s': 1, 'Baby': 2, 'Kid': 3, 'Senior': 4, 'Teenager': 5}, 'race': {'Caucasian': 0, 'Mongoloid': 1, 'Negroid': 2}, 'masked': {'masked': 0, 'unmasked': 1}, 'skintone': {'dark': 0, 'light': 1, 'mid-dark': 2, 'mid-light': 3}, 'emotion': {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 6}, 'gender': {'Female': 0, 'Male': 1}},
    'train_batch_size': 16,
    'valid_batch_size': 16,
    'test_batch_size': 1,
    'epochs': 10,
    'seed': 42,
    'model_path': 'model/model.bin',
    'model_vit_path': 'model/vit_50_epoch.pth',
    'yolo_path': 'model/yolov8l-face.pt',
    'num_workers': os.cpu_count(),
    'd_model': 512,
    'n_layers': 6,
    'heads': 8,
    'dropout': 0.1,
    'max_strlen': 100,
    'k': 5
}
