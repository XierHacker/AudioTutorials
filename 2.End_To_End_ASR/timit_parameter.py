import sys
sys.path.append("/../")
sys.path.append("/../../")
from Utils import statistic
#from Utility import embed

#-------------------------------------------参数列表----------------------------------------------------#
HIDDEN_UNITS_NUM=256                #隐藏层结点数量
HIDDEN_UNITS_NUM2=256               #隐藏层2结点数量

MAX_LABEL_SIZE=80                   #label序列的最大长度为80
MAX_FRAME_SIZE=520                  #语音特征帧的最大长度
MFCC_FEATURES=39                    #mfcc的特征维度
CLASS_NUM=28+1                      #通过字母表示的类别数量，28+"BLINK"

#字向量样本数和维度信息
# CHAR_EMBEDDING=embed.readEmbeddings(file="../../../Embeddings/char_vec.txt")
# WORD_EMBEDDING=embed.readEmbeddings(file="../../../Embeddings/word_vec.txt")
# CHAR_VOCAB_SIZE,CHAR_EMBEDDING_SIZE=statistic.CharVecInfo(char_vec_file="../../../Embeddings/char_vec.txt")
# WORD_VOCAB_SIZE,WORD_EMBEDDING_SIZE=statistic.CharVecInfo(char_vec_file="../../../Embeddings/word_vec.txt")
#INPUT_SIZE=CHAR_EMBEDDING_SIZE                     #嵌入字向量维度,和输入大小应当一样

LEARNING_RATE=0.001                 #学习率
DECAY_RATE=0.9                      #学习率衰减
MAX_EPOCH=15                        #最大迭代次数
BATCH_SIZE=30                       #batch大小
KEEP_PROB=0.8                       #dropout过程保存的比率


# 训练集样本数目
TRAIN_FILE_LIST=["../DataSet/timit_train.tfrecords"]
TRAIN_SIZE=statistic.getTFRecordsListAmount(tfFileList=TRAIN_FILE_LIST)

#测试集样本数目
TEST_FILE_LIST=["../DataSet/timit_test.tfrecords"]
TEST_SIZE=statistic.getTFRecordsListAmount(tfFileList=TEST_FILE_LIST)

MODEL_SAVING_DIR="./saved_models/epoch_"            #模型存储目录

#----------------------------------------------------------------------------------------------------#

if __name__=="__main__":
    print("train_file_list:",TRAIN_FILE_LIST)
    print("train_size:",TRAIN_SIZE)

    print("test_file_list:", TEST_FILE_LIST)
    print("test_size:", TEST_SIZE)

    # print(CHAR_VOCAB_SIZE,CHAR_EMBEDDING_SIZE)
    # print(WORD_VOCAB_SIZE, WORD_EMBEDDING_SIZE)
    # print(CHAR_EMBEDDING.shape)
    # print(WORD_EMBEDDING.shape)