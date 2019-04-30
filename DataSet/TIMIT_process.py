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
    for top_dir,dirs,files in result:
        print("top_dir:",top_dir)
        print("dirs:",dirs)
        print("files:",files)
        print("\n\n")
        if len(files)==0:
            continue
        for file in files:
            file_list=os.path.splitext(file)
            #print("file_list:",file_list)
            #.WAV
            if ".WAV" in file_list:
                #去掉SA开头的
                if "SA" in file_list[0]:
                    continue
                print("file_list:", file_list)
                num_samples+=1
    print("num_samples:",num_samples)
    #test_dir=os.path.join(save_dir,"1","test","face")
    # print("test_dir:",test_dir)



if __name__=="__main__":
    preprocess(ROOT_DIR,SAVE_DIR,False)