import os
import time
import sys
sys.path.append("../")
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import librosa
import tensorflow as tf
from Utils import dst            #feature extractor
from Utils import index_utils
import warnings
warnings.filterwarnings("ignore")

NUM_WORKERS=4
ROOT_DIR=r"D:\WorkSpace\data\TIMIT"
SAVE_DIR=r"D:\WorkSpace\data\TIMIT\Processed"
MAX_FRAME_SIZE=520
MAX_LABEL_SIZE=80
MFCC_FEATURES=39

IS_WRITE=True           #是否写入tfrecords

def get_features(audioFile,labelFile,max_frame_size=MAX_FRAME_SIZE,max_label_size=MAX_LABEL_SIZE):
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
    # print("mfcc\n", mfcc.shape)
    # padding label index list
    mfcc_padded = np.zeros(shape=(max_frame_size,mfcc.shape[1]), dtype=mfcc.dtype)
    mfcc_padded[:mfcc.shape[0]]=mfcc[:]
    #print("mfcc_padded:\n", mfcc_padded)
    features.append(mfcc_padded)                # 记录mfcc结果
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
    #padding label index list
    index_list_padded = np.zeros(shape=(max_label_size,), dtype=type(index_list[0]))
    index_list_padded[:len(index_list)] = index_list[:]
    #print("index_list_padded:", index_list_padded)
    features.append(index_list_padded)         # 记录padding之后的索引列表结果
    features.append(len(index_list))            # 记录索列表真实长度
    return features


def preprocess(root_dir,save_dir,is_train=True,max_workers=None):
    '''
    预处理函数，把root_dir下面的所有音频找到提取特征并转化为tfrecords
    :param root_dir:TIMIT根目录
    :param save_dir:tfrecords保存目录
    :param is_train:是否是训练集
    :param max_workers:最大CPU使用数量
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

    #多进程处理得到特征
    start_time=time.time()
    executor=ProcessPoolExecutor(max_workers=max_workers)
    futures=[]
    index=1
    for file in process_files:
        #if index>10:
        #    break
        audio_file=file+".WAV"
        label_file=file+".WRD"
        # print("audio_file:",audio_file)
        # print("labels_file:",label_file)
        futures.append(executor.submit(get_features,audio_file,label_file))
        index+=1
    records=[future.result() for future in futures]
    end_time=time.time()

    #print(len(records))
    #print("record[0]",records[0])
    print("spend: ", end_time - start_time, " s")
    #infomation
    print("------------information--------------------")
    max_audio=0
    max_label=0
    for record in records:
        if record[1]>max_audio:
            max_audio=record[1]
        if record[3]>max_label:
            max_label=record[3]
    print("max_audio:",max_audio)
    print("max_labels:",max_label)
    print("-------------------------------------------")

    #写入到tfrecords
    writer=tf.python_io.TFRecordWriter(path="./timit.tfrecords")
    for record in records:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "mfcc":tf.train.Feature(float_list=tf.train.FloatList(value=record[0].flatten())),
                    "mfcc_len":tf.train.Feature(int64_list=tf.train.Int64List(value=[record[1]])),
                    "label":tf.train.Feature(int64_list=tf.train.Int64List(value=record[2])),
                    "label_len":tf.train.Feature(int64_list=tf.train.Int64List(value=[record[3]]))
                }
            )
        )
        writer.write(record=example.SerializeToString())


def _parse_data(example_proto):
    '''
    定义tfrecords解析和预处理函数
    :param example_proto:
    :return:
    '''
    parsed_features = tf.parse_single_example(
        serialized=example_proto,
        features={
            "mfcc":tf.FixedLenFeature(shape=[MAX_FRAME_SIZE*MFCC_FEATURES,], dtype=tf.float32),
            "mfcc_len":tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "label":tf.FixedLenFeature(shape=[80, ], dtype=tf.int64),
            "label_len":tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }
    )
    mfcc=tf.reshape(parsed_features["mfcc"],shape=(MAX_FRAME_SIZE,MFCC_FEATURES))
    mfcc_len=tf.cast(parsed_features["mfcc_len"],tf.int32)
    label=tf.cast(parsed_features["label"],tf.int32)
    label_len=tf.cast(parsed_features["label_len"],tf.int32)
    return mfcc,mfcc_len,label,label_len


def readTFRecords(tfrecords_file_list):
    '''
    读取tfrecords中的内容
    :param inFile: tfrecords
    :return:
    '''
    # ----------------------------------------data set API-----------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_file_list)
    # 使用map处理得到新的dataset
    dataset = dataset.map(map_func=_parse_data)
    dataset = dataset.batch(1)

    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # ---------------------------------------------------------------------------------------------

    with tf.Session() as sess:
        for i in range(10):
            print("Sample:", i)
            mfcc,mfcc_len,label,label_len = sess.run(next_element)
            print("mfcc:\n",mfcc)
            print("mfcc.shape",mfcc.shape)
            print("\n")
            print("mfcc_len:\n", mfcc_len)
            print("\n")
            print("label:\n", label)
            print("\n")
            print("label_len:\n", label_len)
            print("\n")



if __name__=="__main__":
    if IS_WRITE:
        preprocess(ROOT_DIR,SAVE_DIR,True,NUM_WORKERS)
    else:
        readTFRecords(tfrecords_file_list=["./timit.tfrecords"])
