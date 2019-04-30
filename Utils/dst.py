import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import preprocessing


def pre_emphasis(x,alpha=0.95):
    '''
    对于原始波形进行预加重x'(t)=x(t)-alpha*x(t-1)
    :param x: np.ndarray;表示输入信号，其中每个元素的范围应该是[-1, 1].
    :param alpha: 预加重的系数，范围为[0.95,0.99],默认是0.97
    :return:
    '''
    return np.append(x[0],x[1:]-alpha*x[:-1])


def MFCC_Delta(audio,sample_rate,alpha=0.97,n_fft=512,win_length=0.025,win_step=0.01,n_mels=26,n_mfcc=13):
    '''
    13维度MFCC+13维度一阶差分+13维度二阶差分
    :param audio:
    :param sample_rate:
    :param alpha:
    :param n_fft:
    :param win_length:
    :param win_step:
    :param n_mels:
    :param n_mfcc:
    :return:
    '''
    #step 1:preemphasis
    audio=pre_emphasis(x=audio,alpha=alpha)
    #step 2:stft
    win_length_samples=int(sample_rate*win_length)          #窗长时间转窗采样点数量
    win_step_samples=int(sample_rate*win_step)              #帧移时间转帧移采样点数量
    win_hop_samples=win_length_samples-win_step_samples     #两帧重叠部分
    # print("win_length_samples:",win_length_samples)
    # print("win_step_samples:",win_step_samples)
    # print("win_hop_sample:",win_hop_samples)
    if win_length_samples>n_fft:
        n_fft=win_length_samples
    stft_feature=librosa.core.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=win_hop_samples,
        win_length=win_length_samples
    )
    print(np.abs(stft_feature).shape)
    #step2 mel
    mel=librosa.feature.melspectrogram(S=np.abs(stft_feature)**2,n_mels=n_mels)
    # print("mel:\n",mel.shape)
    #step3 mfcc
    mfcc=librosa.feature.mfcc(S=librosa.power_to_db(mel),n_mfcc=n_mfcc)
    #print("mfcc:\n",mfcc)
    #step4 delta-1 and delta-2
    delta_1=librosa.feature.delta(data=mfcc,width=3,order=1)
    #print("delta_1:",delta_1)
    delta_2=librosa.feature.delta(data=mfcc,width=3,order=2)
    #print("delta_2:",delta_2)
    features=np.transpose(np.concatenate((mfcc,delta_1,delta_2),0))
    features=preprocessing.scale(X=features)
    #print("features:", features)
    return features


# def mel_fbank():
#     eps = np.spacing(1)
#     magnitude_spectrogram = np.abs(librosa.stft(signal + eps,
#                                                 n_fft=512,
#                                                 hop_length=384,
#                                                 center=True)) ** 2  # np.shape(magnitude_spectrogram)
#     # signal = read_pcm(pcm_path)
#     mel_basis = librosa.filters.mel(sr=16000,
#                                     n_fft=512,
#                                     n_mels=39)  # np.shape(mel_basis)控制输出个数
#     mel_spectrum = np.dot(mel_basis, magnitude_spectrogram)
#     logenergy = np.log(np.sum(magnitude_spectrogram, axis=0)).reshape(1, mel_spectrum.shape[1])
#     S = np.concatenate((librosa.logamplitude(mel_spectrum), logenergy), axis=0)  # np.shape(S)


if __name__=="__main__":
    audio, rate = librosa.core.load(path="../res/1.wav", sr=None)
    print("audio:\n", audio)
    print("rate:\n", rate)
    MFCC_Delta(audio=audio,sample_rate=rate)