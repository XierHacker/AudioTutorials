import os
import sys
sys.path.append("../")
import time
import numpy as np
import tensorflow as tf
import model
import timit_parameter
from Utils import index_utils

#指定显卡
#os.environ['CUDA_VISIBLE_DEVICES']='0'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

#不提示调试信息和警告信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

GRAPH_DEF_SAVING_DIR="./checkpoints/"
GRAPH_DEF_SAVING_NAME="lstm.pbtxt"
MODEL_SAVING_PATH="./checkpoints/lstm.ckpt"

def _parse_data(example_proto):
    '''
    定义tfrecords解析和预处理函数
    :param example_proto:
    :return:
    '''
    parsed_features = tf.parse_single_example(
        serialized=example_proto,
        features={
            "mfcc":tf.FixedLenFeature(shape=[timit_parameter.MAX_FRAME_SIZE*timit_parameter.MFCC_FEATURES,], dtype=tf.float32),
            "mfcc_len":tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "label":tf.FixedLenFeature(shape=[80, ], dtype=tf.int64),
            "label_len":tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }
    )
    mfcc=tf.reshape(parsed_features["mfcc"],shape=(timit_parameter.MAX_FRAME_SIZE,timit_parameter.MFCC_FEATURES))
    mfcc_len=tf.cast(parsed_features["mfcc_len"],tf.int32)
    label=tf.cast(parsed_features["label"],tf.int32)
    label_len=tf.cast(parsed_features["label_len"],tf.int32)
    return mfcc,mfcc_len,label,label_len

#传入的是一个tfrecords文件列表
def train(tfrecords_file_list):
    # ----------------------------------------data set API-----------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_file_list)
    # 使用map映射的处理函数处理得到新的dataset
    dataset = dataset.map(map_func=_parse_data,num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size=1000).batch(timit_parameter.BATCH_SIZE).repeat()

    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # ---------------------------------------------------------------------------------------------

    # 序列长度
    mfcc_len=next_element[1]
    mfcc_mask = tf.sequence_mask(
        lengths=mfcc_len,
        maxlen=timit_parameter.MAX_FRAME_SIZE,
        name="mfcc_mask"
    )               # 用来去掉在mfcc特征帧上padding的mask
    label_len=next_element[3]
    label_mask = tf.sequence_mask(
        lengths=label_len,
        maxlen=timit_parameter.MAX_LABEL_SIZE,
        name="label_mask"
    )               # 用来去掉在label序列上padding的mask

    #输入和标签
    mfcc=next_element[0]
    label=next_element[2]
    # label_hot = tf.one_hot(indices=label, depth=timit_parameter.CLASS_NUM)  # one-hot转换[batch_size,max_label_size,class_num]
    # print("shape of y_hot:", y_hot)
    # y_masked = tf.boolean_mask(tensor=y, mask=mask, name="y_p_masked")  #去掉padding[seq_len1+seq_len2+....+lenN,]
    # y_hot_masked = tf.boolean_mask(tensor=y_hot, mask=mask,name="y_hot_p_masked") # [seq_len1+seq_len2+....lenN,class_num]

    #keep prob
    keep_prob=timit_parameter.KEEP_PROB

    #
    # char_embedings=tf.constant(value=parameter.CHAR_EMBEDDING,dtype=tf.float32,name="char_embeddings")
    # print("char_embeddings.shape", char_embedings.shape)
    # 使用regularizer控制权重
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    acoustic_model=model.AcousticModel()

    #--------------------------------------------------logits------------------------------------------------------
    logits=acoustic_model.forward(mfcc,mfcc_len,keep_prob,False)
    print("logits.shape:",logits.shape)
    #logits_normal [parameter.BATCH_SIZE,max_time_stpes,parameter.CLASS_NUM]
    # logits_normal = tf.reshape(logits,(-1, parameter.MAX_SENTENCE_SIZE, parameter.CLASS_NUM),"logits_normal")
    #logits_masked [seq_len1+seq_len2+..+seq_lenn, 5]
    # logits_masked = tf.boolean_mask(tensor=logits_normal,mask=mask,name="logits_masked")
    # print("logits_masked.shape", logits_masked.shape)
    #---------------------------------------------------------------------------------------------------------------

    #----------------------------------------------------CTC Loss---------------------------------------------------
    negative_log_probability=tf.nn.ctc_loss_v2(
        labels=label,
        logits=logits,
        label_length=label_len,
        logit_length=mfcc_len,
        logits_time_major=False,
        blank_index=-1,
        name="ctc_loss"
    )
    #----------------------------------------------------------------------------------------------------------------

    #------------------------------------------------ prediction------------------------------------------------------
    result=tf.nn.ctc_beam_search_decoder_v2(
        inputs=tf.transpose(logits,(1,0,2)),
        sequence_length=mfcc_len
    )[0][0]
    result_dense=tf.sparse_tensor_to_dense(sp_input=result)
    #pred = tf.cast(tf.argmax(logits, 1), tf.int32, name="pred")  #[parameter.BATCH_SIZE*max_time, 1]
    #pred_normal = tf.reshape(tensor=pred,shape=(-1, parameter.MAX_SENTENCE_SIZE),name="pred_normal") #[parameter.BATCH_SIZE, max_time]
    # pred_normal = tf.reshape(tensor=decode_tags, shape=(-1, parameter.MAX_SENTENCE_SIZE), name="pred_normal")
    # pred_masked = tf.boolean_mask(tensor=pred_normal, mask=mask, name="pred_masked")  # [seq_len1+seq_len2+....+,]
    #---------------------------------------------------------------------------------------------------------------


    # #----------------------------------------------CRF--------------------------------------------------------------
    # log_likelihood,transition_params=tf.contrib.crf.crf_log_likelihood(
    #     inputs=logits_normal,tag_indices=y,sequence_lengths=seq_len
    # )
    #
    # # decode,potentials:[batch_size, max_seq_len, num_tags]  decode_tags:[batch_size, max_seq_len]
    # decode_tags, best_score = tf.contrib.crf.crf_decode(
    #     potentials=logits_normal,transition_params=transition_params,sequence_length=seq_len
    # )
    # #---------------------------------------------------------------------------------------------------------------
    #

    #
    # # accracy
    # correct_prediction = tf.equal(pred_masked, y_masked)
    # accuracy = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32),name="accuracy")
    #
    # loss
    l2_loss = tf.losses.get_regularization_loss()
    loss =tf.reduce_mean(negative_log_probability)+l2_loss
    #
    # 学习率衰减
    global_step = tf.Variable(initial_value=1, trainable=False)
    start_learning_rate = timit_parameter.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(
        learning_rate=start_learning_rate,
        global_step=global_step,
        decay_steps=(timit_parameter.TRAIN_SIZE // timit_parameter.BATCH_SIZE) + 1,
        decay_rate=timit_parameter.DECAY_RATE,
        staircase=True,
        name="decay_learning_rate"
    )

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    init_op = tf.global_variables_initializer()
    #
    #saver
    saver=tf.train.Saver()
    #
    # ------------------------------------Session-----------------------------------------
    with tf.Session(config=config) as sess:
        sess.run(init_op)  # initialize all variables
        # save models
        if not os.path.exists(GRAPH_DEF_SAVING_DIR):
            os.mkdir(GRAPH_DEF_SAVING_DIR)
        # save praphdef
        print("save graph def.....")
        tf.train.write_graph(
            graph_or_graph_def=sess.graph_def,
            logdir=GRAPH_DEF_SAVING_DIR,
            name=GRAPH_DEF_SAVING_NAME,
            as_text=True
        )
        for epoch in range(1, timit_parameter.MAX_EPOCH + 1):
            print("Epoch:", epoch)
            # time evaluation
            start_time = time.time()
            train_losses = [];
            train_accus = []  # training loss/accuracy in every mini-batch
            # mini batch
            for i in range(0, (timit_parameter.TRAIN_SIZE // timit_parameter.BATCH_SIZE)):
                # mfcc, mfcc_len, label, label_len = sess.run(next_element)
                # print("mfcc:\n", mfcc)
                # print("mfcc.shape", mfcc.shape)
                # print("\n")
                # print("mfcc_len:\n", mfcc_len)
                # print("\n")
                # print("label:\n", label)
                # print("\n")
                # print("label_len:\n", label_len)
                # print("\n")
                train_loss, result_,label_,_ = sess.run(fetches=[loss, result_dense,label,optimizer], )
                print("train_loss:", train_loss)
                print("result_:\n", result_)
                recover(result=result_,label=label_)

                # add to list,
                train_losses.append(train_loss);
                # train_accus.append(train_accuracy)
            end_time = time.time()
            print("spend: ", (end_time - start_time) / 60, " mins")
            print("average train loss:",sum(train_losses)/len(train_losses))
            # print("average train accuracy:",sum(train_accus)/len(train_accus))
            print("model saving....")
            saver.save(sess=sess, save_path=MODEL_SAVING_PATH, global_step=epoch)
            print("model saving done!")


def recover(result,label):
    '''
    :param result:
    :return:
    '''
    key2id, id2key=index_utils.get_mapper(index_file="../IndexFiles/en_char.csv")
    # print("key2id:\n",key2id)
    recoverd=[]
    print("recoverd:\n")
    for i in range(result.shape[0]):
        s=""
        for j in range(result.shape[1]):
            if int(result[i][j])==0:
                s+=" "
            else:
                s+=id2key[int(result[i][j])]
        print("s:",s)


    print("label:\n")
    for i in range(label.shape[0]):
        s=""
        for j in range(label.shape[1]):
            if int(label[i][j])==0:
                s+=" "
            else:
                s+=id2key[int(label[i][j])]
        print("s:",s)





if __name__=="__main__":
    train(tfrecords_file_list=timit_parameter.TRAIN_FILE_LIST)
