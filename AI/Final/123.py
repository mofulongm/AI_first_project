from moviepy.editor import *
from pydub import AudioSegment


videoFile = 'C:/Users/15339/Desktop/AI/Emotion/mv/tv4.mp4'  # 视频文件
video = VideoFileClip(videoFile)
print(video.end)
a=video.audio.volumex(15)
audio1 = a

print(video.end)
b=AudioFileClip('1869014829.mp3')
audio2 = b.subclip(0,video.end)
audiocct = CompositeAudioClip([audio1,audio2])
videos = video.set_audio(audiocct)  # 音频文件
videos.write_videofile('output1.mp4', audio_codec='aac')  # 保存合成视频，注意加上参数audio_codec='aac'，否则音频无声音