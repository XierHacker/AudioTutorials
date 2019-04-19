import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

def load_test():
    audio,rate=librosa.core.load(path="../res/1.wav",sr=None)
    print("audio.shape:",audio.shape)
    print("rate:",rate)
    print("audio:\n",audio)

    #get duration
    print("duration directly from file:",librosa.get_duration(filename="../res/1.wav"))
    print("duration from y:",librosa.get_duration(y=audio))


    #draw
    x=[i for i in range(audio.shape[0])]
    #print("x:",x)
    plt.plot(x,audio)
    plt.show()



def write_test():
    pass


if __name__=="__main__":
    load_test()