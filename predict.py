import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from config import opt

def apply_softmax(predictions, dim=1):
    return torch.nn.functional.softmax(predictions, dim=dim)

def apply_argmax(probabilities):
    return torch.argmax(probabilities, dim=1).cpu().numpy()[0]

def predict(data, model, device):
    model.eval()

    with torch.no_grad():
        file_name = data["file_name"]
        image = data["image"].to(device)
        trg_age = data["trg_age"].to(device)
        trg_race = data["trg_race"].to(device)
        trg_masked = data["trg_masked"].to(device)
        trg_skintone = data["trg_skintone"].to(device)
        trg_emotion = data["trg_emotion"].to(device)
        trg_gender = data["trg_gender"].to(device)

        _, age, race, masked, skintone, emotion, gender, _, _, _ = model(
            file_name=file_name,
            image=image.unsqueeze(0),
            trg_age=trg_age,
            trg_race=trg_race,
            trg_masked=trg_masked,
            trg_skintone=trg_skintone,
            trg_emotion=trg_emotion,
            trg_gender=trg_gender,
        )
        
        # Áp dụng softmax và lấy argmax cho từng lớp
        age_probabilities = apply_softmax(age)
        race_probabilities = apply_softmax(race)
        masked_probabilities = apply_softmax(masked)
        skintone_probabilities = apply_softmax(skintone)
        emotion_probabilities = apply_softmax(emotion)
        gender_probabilities = apply_softmax(gender)

        age_pred = apply_argmax(age_probabilities)
        race_pred = apply_argmax(race_probabilities)
        masked_pred = apply_argmax(masked_probabilities)
        skintone_pred = apply_argmax(skintone_probabilities)
        emotion_pred = apply_argmax(emotion_probabilities)
        gender_pred = apply_argmax(gender_probabilities)

    return file_name, age_pred, race_pred, masked_pred, skintone_pred, emotion_pred, gender_pred, data["image"]

def predict_result(pre):
    column_names = ['file_name', 'age', 'race', 'masked', 'skintone', 'emotion', 'gender', 'image']
    df_pre = pd.DataFrame(pre).transpose()
    df_pre.columns = column_names
    for column in df_pre.columns:
        if column != 'file_name' and column != 'image':
            df_pre[column] = df_pre[column].map({v: k for k, v in opt['encoding_mapping'][column].items()})    
    return df_pre