import cv2
import numpy as np
import os
from pyaudio import PyAudio, paInt16 
import numpy as np 
from datetime import datetime 
import wave

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

CUR_PATH = 'C:/Users/15339/Desktop/test'
del_file(CUR_PATH)
cap = cv2.VideoCapture('C:/Users/15339/Desktop/AI/Emotion/mv/tv4.mp4')        # 打开摄像头
i=0
while True:
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