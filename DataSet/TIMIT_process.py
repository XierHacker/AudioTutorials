import os
import sys
sys.path.append("../")
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf
from Utils import dst                       #feature extractor
from Utils import index_utils

ROOT_DIR=r"D:\WorkSpace\data\TIMIT"
SAVE_DIR=r"D:\WorkSpace\data\TIMIT\Processed"

def preprocess(root_dir,save_dir,is_train=True):
    process_dir=""
    if is_train:
        process_dir = os.path.join(root_dir, "TRAIN")
    else:
        process_dir=os.path.join(root_dir,"TEST")
    print("process_dir:",process_dir)
    result=os.walk(top=process_dir)         #iter all structure
    num_samples=0
    mfcc_records=[]           #用来记录所有mfcc结果
    mfcc_seq_len = []         #用来记录每个mfcc的真实帧长度
    labels_records=[]         #用来记录所有label结果
    labels_seq_len=[]         #用来记录每个label的真实序列长度
    key2id,id2key=index_utils.get_mapper(index_file="../IndexFiles/en_char.csv")
    # print("key2id:",key2id)
    # print("id2key:",id2key)
    for top_dir,dirs,files in result:
        # print("top_dir:",top_dir)
        # print("dirs:",dirs)
        # print("files:",files)
        # print("\n\n")
        if len(files)==0:
            continue
        for file in files:
            file_list=os.path.splitext(file)
            #print("file_list:",file_list)
            #.WAV
            if ".WAV" in file_list:
                #去掉SA开头的(SA相关文件影响结果准确性)
                if "SA" in file_list[0]:
                    continue
                audio_file =os.path.join(top_dir,file_list[0])+".WAV"
                label_file =os.path.join(top_dir,file_list[0])+".WRD"
                #print("audio_file:",audio_file)
                #print("label_file:",label_file)
                #提取音频特征
                audio,sample_rate=librosa.core.load(path=audio_file,sr=None)
                # print("audio:",audio)
                # print("sample_rate:",sample_rate)
                mfcc=dst.MFCC_Delta(audio=audio,sample_rate=sample_rate)
                mfcc_records.append(mfcc)           #记录mfcc结果
                mfcc_seq_len.append(mfcc.shape[0])  #记录mfcc真实帧长度
                #print("mfcc.shape",mfcc.shape)

                #得到标签字母/音素特征并且转化为id
                label_in=open(file=label_file)
                lines=label_in.readlines()
                index_list = []
                for line in lines:
                    line_list=line.strip().split(sep=" ")
                    # print("line_list:", line_list)
                    for char in line_list[-1]:
                        index_list.append(key2id[char])
                    index_list.append(0)
                label_in.close()
                index_list.pop()     #去掉最后一个多余0
                # print("index_list:",index_list)
                labels_records.append(index_list)   #记录索引列表结果
                labels_seq_len.append(len(index_list))      #记录索列表真实长度





                #print("file_list:", file_list)
                num_samples+=1
    print("num_samples:",num_samples)
    #test_dir=os.path.join(save_dir,"1","test","face")
    # print("test_dir:",test_dir)


if __name__=="__main__":
    preprocess(ROOT_DIR,SAVE_DIR,False)