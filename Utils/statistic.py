import numpy as np
import tensorflow as tf

def getTFRecordsAmount(tfFile):
    '''
    统计tfrecords中样本数量
    :param tfFile: 相应样本集的tfrecords文件
    :return:    样本总数量
    '''
    num = 0
    for record in tf.python_io.tf_record_iterator(tfFile):
        num += 1
    return num


def getTFRecordsListAmount(tfFileList):
    '''
    统计tfrecords列表中样本数量
    :param tfFile: 相应样本集的tfrecords文件列表
    :return:    样本总数量
    '''

    num = 0
    for file in tfFileList:
        num+=getTFRecordsAmount(tfFile=file)
    return num


def CharVecInfo(char_vec_file):
    '''
    得到字向量样本数和维度
    :param char_vec_file:
    :return:
    '''
    f = open(file=char_vec_file, encoding="utf-8")
    lines = f.readlines()
    # first row is info
    info = lines[0].strip()
    info_list = info.split(sep=" ")
    vocab_size = int(info_list[0])
    embedding_dims = int(info_list[1])
    return vocab_size,embedding_dims

def WordVecInfo(word_vec_file):
    '''
    得到词向量样本数和维度
    :param word_vec_file:
    :return:
    '''
    f = open(file=word_vec_file, encoding="utf-8")
    lines = f.readlines()
    # first row is info
    info = lines[0].strip()
    info_list = info.split(sep=" ")
    vocab_size = int(info_list[0])
    embedding_dims = int(info_list[1])
    return vocab_size,embedding_dims


if __name__=="__main__":
    vocab_size,embedding_dims=CharVecInfo(char_vec_file="../Embeddings/char_vec.txt")
    print("vocab_size:",vocab_size)
    print("embeddings_dim:",embedding_dims)
    tfFileList=["../DataSet/WordSegment/word_seg_"+str(i)+".tfrecords" for i in range(16)]
    print(getTFRecordsListAmount(tfFileList=tfFileList))
