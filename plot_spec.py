import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pydub import AudioSegment
import os
from PIL import Image, ImageOps

#音源のステレオ→モノラル変換
def convert_to_mono(wav_filename,out_filename):
     sound = AudioSegment.from_wav(wav_filename) 
     sound = sound.set_channels(1) 
     sound.export(out_filename, format="wav") 

#スペクトラムとスペクトログラムの画像出力関数
def plot(wav_filename,output_spec_filename,output_melspec_filename):
     #inputデータ読み取り
     y, sr = librosa.load(wav_filename) #波形情報とサンプリングレートを出力

     ##### スペクトログラムを表示する #####
     # フレーム長
     fft_size = 512                 
     # フレームシフト長 
     hop_length = int(fft_size / 4)  

     # 短時間フーリエ変換実行
     amplitude = np.abs(librosa.core.stft(y, n_fft=fft_size, hop_length=hop_length))

     # 振幅をデシベル単位に変換
     log_power = librosa.core.amplitude_to_db(amplitude)

     # グラフ表示
     # librosa.display.specshow(log_power, sr=rate, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
     librosa.display.specshow(log_power, sr=sr, hop_length=hop_length, cmap='magma')
     plt.savefig(output_spec_filename)

     ##### メルスペクトログラムを表示する #####

     # メルスペクトログラム計算
     amplitude_2 = amplitude**2
     log_stft = librosa.power_to_db(amplitude_2)
     melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=64)

     # グラフ表示
     # librosa.display.specshow(melsp, sr=rate, x_axis="time", y_axis="mel", hop_length=hop_length)
     librosa.display.specshow(melsp, sr=sr, hop_length=hop_length)
     plt.savefig(output_melspec_filename) 

#画像の左右反転
def fliplr(input_image,output_image):
     im = Image.open(input_image)
     #左右反転
     Image.fromarray(np.fliplr(im)).save(output_image)
     
#画像の上下反転
def flipud(input_image,output_image):
     im = Image.open(input_image)
     #上下反転
     Image.fromarray(np.flipud(im)).save(output_image)

#画像の180度回転
def flip180(input_image,output_image):
     im = Image.open(input_image)
     #180度回転
     Image.fromarray(np.flip(im, (0, 1))).save(output_image)

#Loopで打音データを画像に変換していく
for i in range(1, 61, 1):
    
    #打音データの定義
    #iron_wav_file='/Users/ab520221/U研_打音判別/打音サンプル/鉄球/iron_' + str(i) + '.wav'
　
　#スペクトログラム画像の定義
    #iron_spec_image='/Users/ab520221/U研_打音判別/データ生成/画像/スペクトラム/iron_spec_' + str(i) + '.png'
　
　#メルスペクトログラム画像の定義
    #iron_melspec_image='/Users/ab520221/U研_打音判別/データ生成/画像/スペクトラム/iron_melspec_' + str(i) + '.png'

　#打音データのステレオtoモノラル変換実施(必要であれば)
    #input1ステレオ音源
    #input2モノラル音源    
    #convert_to_mono(input1,input2)
    
    #打音データからスペクトログラム/メルスペクトログラム画像の作成
    #inoput1 インプット音源(モノラル音源)
    #inoput2 スペクトラム画像
    #inoput3 メルスペクトラム画像
    #plot(input1,input2,input3)
    
    #画像の左右反転
    #input1入力画像
    #input2 左右反転画像    
    #fliplr(input1,input2)
    
    #画像の上下反転
    #input1 入力画像
    #input2 上下反転画像    
    #flipud(input1,input2)

    #画像の180度回転
    #input1 入力画像
    #input2 180度回転画像   
    #flipud(input1,input2)
