import pandas as pd
import csv
import numpy as np
import librosa
import glob 
import matplotlib.pyplot as plt
import librosa.display
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import os, sys, glob, argparse
from tqdm import tqdm
import time, datetime
import pdb, traceback
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import cv2
from moviepy.editor import *
from pydub import AudioSegment
from pydub.utils import make_chunks
import requests
import re
from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.options import Options
from playsound import playsound
import os
import uuid
from ffmpy import FFmpeg
import random

path_video='C:/Users/15339/Desktop/AI/Emotion/mv/tv4.mp4'

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def tv():
    CUR_PATH = 'C:/Users/15339/Desktop/test'
    del_file(CUR_PATH)
    cap = cv2.VideoCapture(0)        # 打开摄像头
    i=0
    while i<=1000:
        ret, frame = cap.read()       # 读摄像头
        cv2.imshow("video", frame)
        dst = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        res=cv2.resize(dst,(48,48),interpolation=cv2.INTER_AREA)

        i=i+1
        if i%30==0:
            cv2.imwrite("C:/Users/15339/Desktop/test/"+str(i)+".png", res) 
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
            break
        
    cap.release()      
    cv2.destroyAllWindows()    # 基本操作

def voice():
    CUR_PATH = 'C:/Users/15339/Desktop/AI/Emotion/data'
    del_file(CUR_PATH)
    video = VideoFileClip(path_video)
    audio = video.audio
    audio.write_audiofile('C:/Users/15339/Desktop/AI/Emotion/data/test.wav')

    myaudio = AudioSegment.from_file("C:/Users/15339/Desktop/AI/Emotion/data/test.wav" , "wav") 
    chunk_length_ms = 3000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    #Export all of the individual chunks as wav files
    path='C:/Users/15339/Desktop/AI/Emotion/data/'
    i=0
    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        i=i+1
        chunk.export(path+chunk_name, format="wav")
    return i

def Speech(a):
    sum=np.zeros(10)
    result1=[]
    doc=open('C:/Users/15339/Desktop/AI/Emotion/4.csv','w',encoding='utf-8')
    csv_writer = csv.writer(doc)
    csv_writer.writerow(["音频","label"])
    for i in range(a-1):
        lb = LabelEncoder()
        path='C:/Users/15339/Desktop/AI/Emotion/data/chunk'+str(i)+'.wav'
        data, sampling_rate = librosa.load(path)


        X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        featurelive = mfccs
        livedf2 = featurelive
        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim= np.expand_dims(livedf2, axis=2)

        json_file = open('C:/Users/15339/Desktop/AI/Speech-Emotion-Analyzer-master/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("C:/Users/15339/Desktop/AI/Speech-Emotion-Analyzer-master/saved_models/Emotion_Voice_Detection_Model.h5")

        livepreds = loaded_model.predict(twodim, 
                                batch_size=32, 
                                verbose=1)
        livepreds1=livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        data=['female_angry','female_calm','female_fearful','female_happy','female_sad','male_angry','male_calm','male_fearful','male_happy','male_sad']
        data=lb.fit_transform(data)
        # print(liveabc)
        sum[liveabc]+=1
        livepredictions = (lb.inverse_transform((liveabc)))
        csv_writer.writerow(['chunk'+str(i),livepredictions[0]])
        # result1[i].append(livepredictions)
    result=1
    # print(np.argmax(sum))
    doc.close
    print("音频")
    if np.argmax(sum)==4:
        print('female_sad')
    if liveabc[0]==0 or liveabc[0]==2 or liveabc==4 or liveabc==5 or liveabc==7 or liveabc==9:
        result=-1
    return result

class QRDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.img_path[index]).convert('RGB')
        
        lbl_dict = {'angry': 0,
             'disgusted': 1,
             'fearful': 2,
             'happy': 3,
             'neutral': 4,
             'sad': 5,
             'surprised': 6}
        if self.transform is not None:
            img = self.transform(img)
        
        if 'test' in self.img_path[index]:
            return img, torch.from_numpy(np.array(0))
        else:
            lbl_int = lbl_dict[self.img_path[index].split('/')[5].split('\\')[1]]
            #print(lbl_int)
            return img, torch.from_numpy(np.array(lbl_int))
    
    def __len__(self):
        return len(self.img_path)

class ExpressionNet(nn.Module):
    def __init__(self):
        super(ExpressionNet, self).__init__()
                
        model = models.resnet18(True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 7)
        self.resnet = model
         
    def forward(self, img):
        #print(img)     
        out = self.resnet(img) 
        return out

def predict(test_loader, model, tta=10):
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                input = input.to(torch.float32)
                target = target.long()

                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
    
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta


def pictures():
    test_jpg = glob.glob('C:/Users/15339/Desktop/test/*')
    test_jpg = np.array(test_jpg)
    test_jpg.sort()

    test_loader = torch.utils.data.DataLoader(
            QRDataset(test_jpg,
                    transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            ), batch_size=50, shuffle=False, num_workers=0, pin_memory=True
    )
            
    model = ExpressionNet()
    model.load_state_dict(torch.load('C:/Users/15339/Desktop/人脸情绪识别挑战赛/resnet18_fold0.pt'))
    test_pred = predict(test_loader, model, 5)
    # print(len(test_pred))
    for i in range(len(test_pred)):
        for j in range(7):
            if test_pred[i][j]>1e1:
                test_pred[i][j]=test_pred[i][j]*1e-1

    # print(test_pred)
    cls_name = np.array([0,1,3,3,4,5,6])
    submit_df = pd.DataFrame({'name': test_jpg, 'label': cls_name[test_pred.argmax(1)]})
    submit_df['name'] = submit_df['name'].apply(lambda x: x.split('/')[-1])
    # print(submit_df['label'][0])
    sum=np.zeros(7)
    for i in submit_df['label']:
        sum[i]=sum[i]+1
    # print(np.argmax(sum))
    print("画面")
    if np.argmax(sum)==5:
        print('sad')
    if np.argmax(sum)==6 or np.argmax(sum)==3:
        print('happy')
    if np.argmax(sum)==4:
        print('happy')
    if np.argmax(sum)==0:
        print('angry')
    submit_df = submit_df.sort_values(by='name')
    submit_df.to_csv('pytorch_submit_test12.csv', index=None)
    result=1
    if np.argmax(sum)==0 or np.argmax(sum)==1 or np.argmax(sum)==4 or np.argmax(sum)==5:
        result=-1
    return result


def song(a):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    if a==1:
      f=open(r"C:/Users/15339/Desktop/AI/Emotion/happy_song.txt" ,encoding='utf-8', errors='ignore')
    else:
      f=open(r"C:/Users/15339/Desktop/AI/Emotion/sad_song.txt" ,encoding='utf-8', errors='ignore')
    lines=f.readlines()
    sum=[]
    for line in lines:
      sum.append(line)
    name =random.choice(sum)
    
    url_1 = 'https://music.163.com/#/search/m/?s=' + name + '&type=1'

    browser = webdriver.Chrome(executable_path='C:/Users/15339/Desktop/AI/Emotion/chromedriver.exe',chrome_options=chrome_options)
    browser.get(url=url_1)
    browser.switch_to.frame('g_iframe')
    sleep(0.5)
    page_text = browser.execute_script("return document.documentElement.outerHTML")
    browser.quit()

    ex1 = '<a.*?id="([0-9]*?)"'
    ex2 = '<b.*?title="(.*?)"><span class="s-fc7">'
    ex3 = 'class="td w1"><div.*?class="text"><a.*?href=".*?">(.*?)</a></div></div>'

    id_list = re.findall(ex1,page_text,re.M)[::2]
    song_list = re.findall(ex2,page_text,re.M)
    singer_list = re.findall(ex3,page_text,re.M)
    li = list(zip(song_list,singer_list,id_list))

    for i in range(len(li)):
        print(str(i+1) + '.' + str(li[i]),end='\n')

    url_f = 'http://music.163.com/song/media/outer/url?id=' + str(id_list[0]) + '.mp3'

    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36'
    }

    audio_content = requests.get(url=url_f,headers=headers).content

    m_name=str(id_list[0])+'.mp3'
    
    with open(m_name,'wb') as f :
        f.write(audio_content)

    print("爬取成功！！！")
    playsound(m_name)

def background(a):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    if a>=0.5:
      f=open(r"C:/Users/15339/Desktop/AI/Emotion/happy_song.txt" ,encoding='utf-8', errors='ignore')
    else:
      f=open(r"C:/Users/15339/Desktop/AI/Emotion/sad_song.txt" ,encoding='utf-8', errors='ignore')
    lines=f.readlines()
    sum=[]
    for line in lines:
      sum.append(line)
    name =random.choice(sum)
    url_1 = 'https://music.163.com/#/search/m/?s=' + name + '&type=1'

    browser = webdriver.Chrome(executable_path='C:/Users/15339/Desktop/AI/Emotion/chromedriver.exe',chrome_options=chrome_options)
    browser.get(url=url_1)
    browser.switch_to.frame('g_iframe')
    sleep(0.5)
    page_text = browser.execute_script("return document.documentElement.outerHTML")
    browser.quit()

    ex1 = '<a.*?id="([0-9]*?)"'
    ex2 = '<b.*?title="(.*?)"><span class="s-fc7">'
    ex3 = 'class="td w1"><div.*?class="text"><a.*?href=".*?">(.*?)</a></div></div>'

    id_list = re.findall(ex1,page_text,re.M)[::2]
    song_list = re.findall(ex2,page_text,re.M)
    singer_list = re.findall(ex3,page_text,re.M)
    li = list(zip(song_list,singer_list,id_list))

    for i in range(len(li)):
        print(str(i+1) + '.' + str(li[i]),end='\n')

    url_f = 'http://music.163.com/song/media/outer/url?id=' + str(id_list[0]) + '.mp3'

    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36'
    }

    audio_content = requests.get(url=url_f,headers=headers).content

    m_name=str(id_list[0])+'.mp3'
    video = VideoFileClip(path_video)
    print(video.end)
    a=video.audio.volumex(15)
    audio1 = a

    print(video.end)
    b=AudioFileClip(m_name)
    audio2 = b.subclip(0,video.end)
    audiocct = CompositeAudioClip([audio1,audio2])
    videos = video.set_audio(audiocct)  # 音频文件
    videos.write_videofile('output1.mp4', audio_codec='aac')  # 保存合成视频，注意加上参数audio_codec='aac'，否则音频无声音


def Video_analysis():
    a=voice()
    face_label=pictures()
    voice_label=Speech(a)
    label=0.85*face_label+0.15*voice_label
    background(label)


def Real_time_monitoring():
    tv()
    face_label=pictures()
    song(face_label)

if __name__=="__main__":
    # Video_analysis()
    Real_time_monitoring()
    

