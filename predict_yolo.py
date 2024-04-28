import cv2
import pandas as pd
import torch
from torchvision import transforms
from config import opt

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((384,384),antialias=True),
                              ])

def take_xywh(boxes):
    num_face = boxes.xywhn.shape[0]
    bboxes = []
    for i in range(num_face):
        center_x, center_y, width, height = boxes.xywhn[i]

        x = (center_x - width / 2) * boxes.orig_shape[1]
        y = (center_y - height / 2) * boxes.orig_shape[0]
        w = width * boxes.orig_shape[1]
        h = height * boxes.orig_shape[0]
        x=x.cpu().item()
        y=y.cpu().item()
        w=w.cpu().item()
        h=h.cpu().item()
        bbox=[x,y,w,h]
        bboxes.append(bbox)
    return bboxes

# def predict_yolo(model, result):
#     result = next(iter(result))
#     file_name = result.path.split('\\')[-1]
#     if len(result.boxes.xywhn)!=0:
#         bbox = take_xywh(result.boxes)
#         x,y,w,h=bbox
#         x=int(x);y=int(y);w=int(w);h=int(h)
#         face_image = result.orig_img[y:y+h, x:x+w] 
#     else:
#         face_image = result.orig_img
#     img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB) 
#     img = transform(img).to(opt['device'])
#     # plt.imshow(img.permute(1, 2, 0))
#     outputs = model(img.unsqueeze(0))
#     # outputs.shape
#     _, predictions = torch.max(outputs, 1)
#     predictions.item()

#     return file_name, bbox

def predict_yolo(results):
    file_names, bboxs = [], []
    for result in results:
        file_name = result.path.split('/')[-1]
        if len(result.boxes.xywhn)!=0:
            list_box  = take_xywh(result.boxes)
            for x,y,w,h in list_box:
                img_box = [x,y,w,h]
                file_names.append(file_name)
                bboxs.append(img_box)

    return file_names, bboxs

def FaceData(file_name, bbox, path, transform, stage):
    if stage == 'Image':
        #img_path = f'{path}{file_name}'
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x, y, w, h = bbox
        face_image = image[int(y):int(y)+int(h), int(x):int(x)+int(w)]
    else:
        image = cv2.cvtColor(file_name, cv2.COLOR_BGR2RGB)
        x, y, w, h = bbox
        face_image = image[int(y):int(y)+int(h), int(x):int(x)+int(w)]
    
    if transform:
        face_image = transform(face_image)
    
    return {
        "file_name": "image.jpg",
        "image": face_image,
        "trg_age": torch.tensor(1, dtype=torch.float16),
        "trg_race": torch.tensor(1, dtype=torch.float16),
        "trg_masked": torch.tensor(1, dtype=torch.float16),
        "trg_skintone": torch.tensor(1, dtype=torch.float16),
        "trg_emotion": torch.tensor(1, dtype=torch.float16),
        "trg_gender": torch.tensor(1, dtype=torch.float16),
    }