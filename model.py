import warnings
import timm
import torch.nn as nn
from function import loss_fn, acc_fn, f1_fn

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.optimization")

class FaceModel(nn.Module):
    def __init__(self, num_age, num_race, num_masked, num_skintone, num_emotion, num_gender, mode):
        super(FaceModel, self).__init__()
        self.mode = mode
        self.num_age = num_age
        self.num_race = num_race
        self.num_masked = num_masked
        self.num_skintone = num_skintone
        self.num_emotion = num_emotion
        self.num_gender = num_gender
        #self.vgg = InceptionResnetV1(pretrained='vggface2')
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        #self.vit = nn.Sequential(*list(self.vit.children())[:8])
        #self.model = nn.Sequential(*list(self.vgg.children())[:17])

        self.drop = nn.Dropout(0.3)
        n_feature = 1000
        self.out_age = nn.Linear(n_feature, self.num_age)
        self.out_race = nn.Linear(n_feature, self.num_race)
        self.out_masked = nn.Linear(n_feature, self.num_masked)
        self.out_skintone = nn.Linear(n_feature, self.num_skintone)
        self.out_emotion = nn.Linear(n_feature, self.num_emotion)
        self.out_gender = nn.Linear(n_feature, self.num_gender)
    
    def forward(self, file_name, image, trg_age=None, trg_race=None, trg_masked=None, trg_skintone=None, trg_emotion=None, trg_gender=None):
        outputs = self.vit(image)
        #outputs = self.drop(outputs)
        src_age = self.out_age(outputs)
        src_race = self.out_race(outputs)
        src_masked = self.out_masked(outputs)
        src_skintone = self.out_skintone(outputs)
        src_emotion = self.out_emotion(outputs)
        src_gender = self.out_gender(outputs)
        
        if self.mode == 'train':
            loss_age = loss_fn(src_age, trg_age, self.num_age)
            loss_race = loss_fn(src_race, trg_race, self.num_race)
            loss_masked = loss_fn(src_masked, trg_masked, self.num_masked)
            loss_skintone = loss_fn(src_skintone, trg_skintone, self.num_skintone)
            loss_emotion = loss_fn(src_emotion, trg_emotion, self.num_emotion)
            loss_gender = loss_fn(src_gender, trg_gender, self.num_gender)

            loss = (loss_age + loss_race + loss_masked + loss_skintone + loss_emotion + loss_gender) / 6

            acc_age = acc_fn(src_age, trg_age, self.num_age)
            acc_race = acc_fn(src_race, trg_race, self.num_race)
            acc_masked = acc_fn(src_masked, trg_masked, self.num_masked)
            acc_skintone = acc_fn(src_skintone, trg_skintone, self.num_skintone)
            acc_emotion = acc_fn(src_emotion, trg_emotion, self.num_emotion)
            acc_gender = acc_fn(src_gender, trg_gender, self.num_gender)

            acc = (acc_age + acc_race + acc_masked + acc_skintone + acc_emotion + acc_gender) / 6

            f1_age = f1_fn(src_age, trg_age, self.num_age)
            f1_race = f1_fn(src_race, trg_race, self.num_race)
            f1_masked = f1_fn(src_masked, trg_masked, self.num_masked)
            f1_skintone = f1_fn(src_skintone, trg_skintone, self.num_skintone)
            f1_emotion = f1_fn(src_emotion, trg_emotion, self.num_emotion)
            f1_gender = f1_fn(src_gender, trg_gender, self.num_gender)

            f1_score = (f1_age + f1_race  + f1_masked + f1_skintone + f1_emotion + f1_gender) / 6
        else:
            loss, acc, f1_score = None, None, None

        return  file_name, src_age, src_race, src_masked, src_skintone, src_emotion, src_gender, loss, acc, f1_score