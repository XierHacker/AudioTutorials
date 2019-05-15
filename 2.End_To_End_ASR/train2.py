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
            "label":tf.FixedLenFeature(shape=[timit_parameter.MAX_LABEL_SIZE, ], dtype=tf.int64),
            "label_len":tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }
    )
    mfcc=tf.reshape(parsed_features["mfcc"],shape=(timit_parameter.MAX_FRAME_SIZE,timit_parameter.MFCC_FEATURES))
    mfcc_len=tf.cast(parsed_features["mfcc_len"],tf.int32)
    label=tf.cast(parsed_features["label"],tf.int32)
    label_len=tf.cast(parsed_features["label_len"],tf.int32)
    return mfcc,mfcc_len,label,label_len



def dense2sparse(dense,dtype=np.int32):
    '''
    把稠密矩阵转为tensorflow的稀疏矩阵
    :param dense:
    :return:
    '''
    indices=[]
    values=[]
    dense_shape=[dense.shape[0],dense.shape[1]]
    for i in range(dense.shape[0]):
        for j in range(dense.shape[1]):
            if dense[i,j]!=0:
                indices.append([i,j])
                values.append(dense[i,j])
    indices=np.array(indices)
    values=np.array(values)
    dense_shape=np.array(dense_shape)
    # print("indices:\n",indices)
    # print("values:\n",values)
    # print("dense_shape:\n",dense_shape)
    return (indices,values,dense_shape)




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

    # mfcc_placeholder
    mfcc_p = tf.placeholder(
        dtype=tf.float32,
        shape=(None, timit_parameter.MAX_FRAME_SIZE, timit_parameter.MFCC_FEATURES),
        name="mfcc_p"
    )
    mfcc_len_p = tf.placeholder(
        dtype=tf.int32,
        shape=(None,),
        name="mfcc_len_p"
    )

    # label placeholder
    label_p = tf.placeholder(
        dtype=tf.int32,
        shape=(None, timit_parameter.MAX_LABEL_SIZE),
        name="label_p"
    )
    label_sparse_p = tf.sparse_placeholder(
        dtype=tf.int32,
        name="label_sparse_p"
    )
    label_len_p = tf.placeholder(
        dtype=tf.int32,
        shape=(None,),
        name="label_len_p"
    )

    #keep prob
    keep_prob=timit_parameter.KEEP_PROB

    # 使用regularizer控制权重
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)

    #建立模型
    acoustic_model=model.AcousticModel()

    #--------------------------------------------------logits------------------------------------------------------
    logits=acoustic_model.forward(mfcc_p,mfcc_len_p,keep_prob,False)
    print("logits.shape:",logits.shape)
    # trans to time-major order
    logits = tf.transpose(logits, perm=[1, 0, 2])
    print("logits.shape:", logits.shape)
    #---------------------------------------------------------------------------------------------------------------

    #----------------------------------------------------CTC Loss---------------------------------------------------
    negative_log_probability=tf.nn.ctc_loss(
        labels=label_sparse_p,
        inputs=logits,
        sequence_length=mfcc_len_p
    )
    #----------------------------------------------------------------------------------------------------------------

    #------------------------------------------------ prediction------------------------------------------------------
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
        inputs=logits,
        sequence_length=mfcc_len_p
    )
    decoded_dense=tf.sparse_tensor_to_dense(sp_input=decoded[0])

    # accracy/error rate
    edit_distance = tf.edit_distance(
        hypothesis=tf.cast(decoded[0], tf.int32),
        truth=label_sparse_p
    )
    error = tf.reduce_mean(edit_distance)

    #---------------------------------------------------------------------------------------------------------------

    # loss
    l2_loss = tf.losses.get_regularization_loss()
    loss =tf.reduce_mean(negative_log_probability)+l2_loss



    #
    # # 学习率衰减
    #global_step = tf.Variable(initial_value=1, trainable=False)
    # start_learning_rate = timit_parameter.LEARNING_RATE
    # learning_rate = tf.train.exponential_decay(
    #     learning_rate=start_learning_rate,
    #     global_step=global_step,
    #     decay_steps=(timit_parameter.TRAIN_SIZE // timit_parameter.BATCH_SIZE) + 1,
    #     decay_rate=timit_parameter.DECAY_RATE,
    #     staircase=True,
    #     name="decay_learning_rate"
    # )

    # optimizer and gradient clip
    var_trainable_op = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, var_trainable_op), 1.0)
    optimizer = tf.train.AdamOptimizer(timit_parameter.LEARNING_RATE).apply_gradients(zip(grads, var_trainable_op))

    init_op = tf.global_variables_initializer()

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
                mfcc, mfcc_len, label, label_len = sess.run(next_element)
                label_sparse=dense2sparse(label)
                # print("mfcc:\n", mfcc)
                # print("mfcc.shape", mfcc.shape)
                # print("\n")
                # print("mfcc_len:\n", mfcc_len)
                # print("\n")
                # print("label:\n", label)
                # print("\n")
                # print("label_len:\n", label_len)
                # print("\n")
                loss_,decoded_dense_,error_,_=sess.run(
                    fetches=[loss,decoded_dense,error,optimizer],
                    feed_dict={mfcc_p:mfcc,mfcc_len_p:mfcc_len,label_sparse_p:label_sparse,label_len_p:label_len}
                )

                print("loss:", loss_)
                print("error:\n", error_)
                recover(result=decoded_dense_,label=label)
            #
            #     # add to list,
            #     train_losses.append(train_loss);
            #     # train_accus.append(train_accuracy)
            # end_time = time.time()
            # print("spend: ", (end_time - start_time) / 60, " mins")
            # print("average train loss:",sum(train_losses)/len(train_losses))
            # # print("average train accuracy:",sum(train_accus)/len(train_accus))
            # print("model saving....")
            # saver.save(sess=sess, save_path=MODEL_SAVING_PATH, global_step=epoch)
            # print("model saving done!")


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


    # a=[
    #     [
    #         [0.1,0.2,0.3,0.4],
    #         [0.2,0.1,0.4,0.3]
    #     ],
    #     [
    #         [0.8, 0.6, 0.2, 0.1],
    #         [0.2, 0.4, 0.6, 0.4]
    #     ],
    #     [
    #         [0.2, 0.6, 0.9, 0.11],
    #         [0.10, 0.76, 0.44, 0.23]
    #     ],
    # ]
    #
    # a_t=tf.constant(a,dtype=tf.float32)
    # print("a_t.shape:",a_t.shape)
    #
    # b_t=tf.transpose(a,perm=[1,0,2])
    # with tf.Session() as sess:
    #     print("b_t:\n",sess.run(b_t))