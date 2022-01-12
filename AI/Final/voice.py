from moviepy.editor import *
from pydub import AudioSegment
from pydub.utils import make_chunks

video = VideoFileClip('C:/Users/15339/Desktop/AI/Emotion/mv/tv.mp4')
audio = video.audio
audio.write_audiofile('C:/Users/15339/Desktop/AI/Emotion/data/test.wav')

myaudio = AudioSegment.from_file("C:/Users/15339/Desktop/AI/Emotion/data/test.wav" , "wav") 
chunk_length_ms = 3000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files
path='C:/Users/15339/Desktop/AI/Emotion/data/'
for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    print(chunk_name)
    chunk.export(path+chunk_name, format="wav")