import os
import sys
sys.path.append("../")
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf
from Utils import dst            #feature extractor
from Utils import index_utils

NUM_WORKERS=4
ROOT_DIR=r"D:\WorkSpace\data\TIMIT"
SAVE_DIR=r"D:\WorkSpace\data\TIMIT\Processed"

def get_features(audioFile,labelFile=None):
    '''
    从音频文件和标签文件得到提取出来的结果
    :param audioFile:音频文件
    :param labelFile:标签文件
    :return:features，是一个列表，里面的元素分别为[mfcc,mfcc真实长度,labels结果，labels真实长度]
    '''
    features=[]
    key2id, id2key = index_utils.get_mapper(index_file="../IndexFiles/en_char.csv")
    # print("key2id:",key2id)
    # print("id2key:",id2key)
    # 提取音频特征
    audio, sample_rate = librosa.core.load(path=audioFile, sr=None)
    # print("audio:",audio)
    # print("sample_rate:",sample_rate)
    mfcc = dst.MFCC_Delta(audio=audio, sample_rate=sample_rate)
    # print("mfcc.shape", mfcc.shape)
    features.append(mfcc)                # 记录mfcc结果
    features.append(mfcc.shape[0])       # 记录mfcc真实帧长度
    # 得到标签字母/音素特征并且转化为id
    label_in = open(file=labelFile)
    lines = label_in.readlines()
    index_list = []
    for line in lines:
        line_list = line.strip().split(sep=" ")
        # print("line_list:", line_list)
        for char in line_list[-1]:
            index_list.append(key2id[char])
        index_list.append(0)
    label_in.close()
    index_list.pop()  # 去掉最后一个多余0
    # print("index_list:",index_list)
    features.append(index_list)         # 记录索引列表结果
    features.append(len(index_list))    # 记录索列表真实长度
    return features


def preprocess(root_dir,save_dir,is_train=True,max_workers=None):
    '''
    预处理函数，把root_dir下面的所有音频找到提取特征并转化为tfrecords
    :param root_dir:
    :param save_dir:
    :param is_train:
    :param max_workers:
    :return:
    '''
    process_dir=""
    process_files=[]
    num_samples = 0
    if is_train:
        process_dir = os.path.join(root_dir, "TRAIN")
    else:
        process_dir=os.path.join(root_dir,"TEST")
    print("process_dir:",process_dir)
    result=os.walk(top=process_dir)         #iter all structure
    #收集所有待处理文件名
    for top_dir, dirs, files in result:
        # print("top_dir:",top_dir)
        # print("dirs:",dirs)
        # print("files:",files)
        # print("\n\n")
        if len(files) == 0:
            continue
        for file in files:
            file_list = os.path.splitext(file)
            # print("file_list:",file_list)
            # .WAV
            if ".WAV" in file_list:
                # 去掉SA开头的(SA相关文件影响结果准确性)
                if "SA" in file_list[0]:
                    continue
                process_files.append(os.path.join(top_dir, file_list[0]))
                num_samples += 1

    print("num_samples:", num_samples)
    # print("process_files:", process_files)

    #多进程处理
    executor=ProcessPoolExecutor(max_workers=max_workers)
    futures=[]
    index=1
    for file in process_files:
        audio_file=file+".WAV"
        label_file=file+".WRD"
        # print("audio_file:",audio_file)
        # print("labels_file:",label_file)
        futures.append(executor.submit(get_features,audio_file,label_file))

    records=[future.result() for future in futures]
    print(len(records))





if __name__=="__main__":
    preprocess(ROOT_DIR,SAVE_DIR,False,NUM_WORKERS)