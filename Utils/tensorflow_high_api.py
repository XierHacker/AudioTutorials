'''
    封装常见的算法API，提供更加高层的API，方便后面使用
'''

import os
import numpy as np
import tensorflow as tf


class BasicUtil():
    '''
    基本的一些API的高级封装类
    '''
    def __init__(self):
        pass

    def creteWeights(slef,shape, regularizer, name):
        '''
        新建一个权重,initializer这里可以根据需求修改
        :param shape:权重形状
        :param regularizer:约束器
        :return:返回相应的权重
        '''
        weights = tf.get_variable(
            name=name,
            shape=shape,
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling()
        )
        if regularizer != None:
            tf.add_to_collection(name="regularized", value=regularizer(weights))
        return weights

    # pytorch-like
    def linear(self,inputs, units, activation, regularizer, keep_rate, name, reuse):
        '''
        全连接操作，同时进行dropout
        :param inputs: 输入
        :param units:  神经元数量，同时也是输出的维度
        :param activation: 激活函数，要是不是用的话为None
        :param regularizer: 正则项
        :param keep_rate: dropout保持的比率
        :param name: 名称
        :param reuse: 是否reuse
        :return: 全连接运算之后的输出，形状为[batch_size,units]
        '''
        logits_fc = tf.layers.dense(
            inputs=inputs,
            units=units,
            activation=activation,
            use_bias=True,
            kernel_initializer=tf.initializers.variance_scaling(),
            bias_initializer=tf.initializers.variance_scaling(),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            activity_regularizer=None,
            trainable=True,
            name=name,
            reuse=reuse
        )
        # dropout
        logits_fc = tf.layers.dropout(inputs=logits_fc, rate=1 - keep_rate)
        return logits_fc


    def feed_forward(self,inputs,units,name):
        '''
        feed forward层，实现参考《Attention Is All You Need》。
        包含两个linear层：
            第一层有一个relu激活，第二层没有,
        注意：其中第二层的结点数量我们强制指定为inputs的通道数，便于保持一致
        :param inputs:输入,形状为(batcn,length,channels)
        :param unit:第一层的结点数量
        :param name:scope名
        :return:feed_forward层之后的结果，形状为(batcn,length,channels)
        '''
        with tf.variable_scope(name):
            d=inputs.get_shape().as_list()[-1]  #channel size
            outputs=tf.layers.dense(inputs=inputs,units=units,activation=tf.nn.relu)  #linear 1
            outputs=tf.layers.dense(inputs=outputs,units=d,activation=None)      #linear 2
        return outputs


    def position_encoding(self,inputs,seq_len,max_len,name):
        '''
        进行位置编码,实现参考《Attention Is All You Need》 3.5
        :param inputs:输入，形状为(batch,length,channels)
        :param seq_len:每个句子的真实长度，形状为(batch，)
        :param max_len:最大长度
        :param name:scope名
        :return:位置编码，形状为(batch,length,channels)

          容易理解的代码（但是不能够写进模型）
        代码1：
            result_1=tf.expand_dims(tf.range(length), 0)
            result_2=tf.tile(result_1,[batch_size,1])

            PE=np.array(
                [
                    [pos/np.power(10000,(i-i%2)/d_model) for i in range(d_model)]
                    for pos in range(max_len)
                ]
            )
            PE[:, 0::2] = np.sin(PE[:, 0::2])  # dim 2i
            PE[:, 1::2] = np.cos(PE[:, 1::2])  # dim 2i+1
            #PE=tf.convert_to_tensor(PE,tf.float32)
            #outputs=tf.nn.embedding_lookup(PE,result_2)
            #mask
            #outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
        代码2：
            # #---------步骤一：计算基本的PE
            # pos = tf.cast(tf.tile(tf.expand_dims(tf.range(length), axis=1), [1, d_model]), tf.float32)  # pos
            # i = tf.cast(tf.tile(tf.expand_dims(tf.range(d_model), axis=0), [length, 1]), tf.float32)  # i
            # c = tf.cast(tf.fill([length, d_model], 10000), tf.float32)  # constant value,eg.10000
            # PE = pos / tf.math.pow(c, (i - i % 2) / d_model)
            # #---------步骤二：对PE的channels进行奇数和偶数通道的变换
            # even = tf.math.sin(PE[:, 0::2])  # dim 2i
            # even=tf.transpose(even)
            # even_index=tf.reshape(tf.range(start=0,limit=length+1,delta=2),shape=(-1,1))
            # odd = tf.math.cos(PE[:, 1::2])  # dim 2i+1
            # odd=tf.transpose(odd)
            # odd_index=tf.reshape(tf.range(start=1,limit=length+1,delta=2),shape=(-1,1))
            # #---------步骤三：变换结果修改写入（因为tf不支持原地修改tensor，这里需要一些技巧）
            # PE = tf.transpose(PE)       #转置PE
            # PE=tf.Variable(initial_value=PE,trainable=False)    #做成变量
            # PE_update_even=tf.scatter_nd_update(PE,even_index,even) #修改偶数channels
            # PE_update = tf.scatter_nd_update(PE_update_even, odd_index, odd)    #修改奇数channels
            # PE_update=tf.transpose(PE_update)       #转置回来
            # #---------步骤四：batch扩增
            # PE_batch=tf.cast(tf.tile(tf.expand_dims(PE_update, axis=0), [batch_size, 1,1]), tf.float32)
            # #---------步骤五：mask
            # #max_len_batch=tf.fill(dims=(batch_size,),value=max_len)
            # #mask_len=max_len_batch-seq_len
        #return PE_batch

         注意：
            下面的代码中，pos和i使用了一些tensorflow的小技巧得到两个索引矩阵，方便我们直接使用矩阵进行运算
         其中：
            pos index,类似如下所示
                    [   [0 0 0 0]
                        [1 1 1 1]
                        [2 2 2 2]   ]
            i index,类似如下所示
                    [   [0 1 2 3]
                        [0 1 2 3]
                        [0 1 2 3]   ]

        init =tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print("d_model:",d_model)
            batch_size=sess.run(batch_size)
            print("batch_size:",batch_size)
            length=sess.run(length)
            print("length:",length)

            PE = sess.run(PE)
            print("PE:\n", PE)

            even = sess.run(even)
            print("even:\n", even)
            even_index = sess.run(even_index)
            print("even_index:\n", even_index)

            odd = sess.run(odd)
            print("odd:\n", odd)
            odd_index = sess.run(odd_index)
            print("odd_index:\n", odd_index)

            PE_update = sess.run(PE_update)
            print("PE_update:\n", PE_update)

            PE_batch = sess.run(PE_batch)
            print("PE_batch:\n", PE_batch)

            max_len_batch = sess.run(max_len_batch)
            print("max_len_batch:\n", max_len_batch)

            mask_len = sess.run(mask_len)
            print("mask_len_batch:\n", mask_len)

        '''
        with tf.variable_scope(name):
            d_model = inputs.get_shape().as_list()[-1]   #channel size
            batch_size,length=tf.shape(inputs)[0],tf.shape(inputs)[1]   #批大小和序列长度
            result_1 = tf.expand_dims(tf.range(length), 0)
            result_2 = tf.tile(result_1, [batch_size, 1])
            PE = np.array(
                [
                    [pos / np.power(10000, (i - i % 2) / d_model) for i in range(d_model)]
                    for pos in range(max_len)
                ]
            )
            PE[:, 0::2] = np.sin(PE[:, 0::2])  # dim 2i
            PE[:, 1::2] = np.cos(PE[:, 1::2])  # dim 2i+1
            PE=tf.convert_to_tensor(PE,tf.float32)
            outputs=tf.nn.embedding_lookup(PE,result_2)
            # mask
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
        return outputs


    def label_smoothing(self):
        pass



class RnnUtil():
    '''
    RNN相关算法的一些基本封装
    '''
    def __init__(self):
        pass

    def createLSTMCell(self,num_units, keep_prob, name, reuse=False):
        '''
        创建一个LSTM cell
        :param num_units:cell中units的多少，这个值决定了输出的维度
        :param keep_prob: dropout过程保存的比率
        :param name: 名称
        :param reuse: 是否reuse
        :return: 返回一个高级版本的LSTM cell
        '''
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units=num_units,
            use_peepholes=True,
            cell_clip=None,
            initializer=tf.initializers.variance_scaling(),
            num_proj=None,
            proj_clip=None,
            forget_bias=1.0,
            state_is_tuple=True,
            reuse=reuse,
            name=name
        )
        # residual connection
        # cell=tf.nn.rnn_cell.ResidualWrapper(cell=cell)
        # dropout
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=cell,
            input_keep_prob=1.0,
            output_keep_prob=keep_prob,
            state_keep_prob=1.0
        )
        return cell

    def LSTM_Encoder(self,inputs, cell_forward, cell_backward, seq_len, scope_name):
        '''
        双层LSTM做encoder,直接得到结果outputs和states
        outputs表示所有时间步的输出，对于双向网络，形式为
            前向结果：shape=(30, 80, 128)， 反向结果：shape=(30, 80, 128)
        states表示最后一步每一层的输出，形式为：
            前向：（LSTMStateTuple1,LSTMStateTuple2)
            反向：（LSTMStateTuple1,LSTMStateTuple2)
        :param inputs: 输入
        :param cell_forward:前向LSTM cell
        :param cell_backward: 反向LSTM cell
        :param seq_len: 真实序列长度
        :param scope_name: scopr 名称
        :return: encoder之后的结果,所有时间的encoder_outputs和最终时刻的encoder_status
        '''
        # outputs,states=tf.nn.bidirectional_dynamic_rnn()
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_forward,
            cell_bw=cell_backward,
            inputs=inputs,
            sequence_length=seq_len,
            initial_state_fw=None,
            initial_state_bw=None,
            dtype=tf.float32,
            scope=scope_name
        )
        # print("outputs:",outputs)
        # print("states:",states)
        # 得到前向和反向outputs，并且拼起来
        outputs_forward = outputs[0]  # shape of h is [batch_size, max_time, projec_num]
        outputs_backward = outputs[1]  # shape of h is [batch_size, max_time, projec_num]
        encoder_outputs = tf.concat(values=[outputs_forward, outputs_backward],axis=2)  # [batch_size, max_time, projec_num*2]

        # 得到前向和反向states,因为两层以上，所以取最后一个[-1]
        states_forward = states[0][-1]  # .c:[batch_size,num_units]   .h:[batch_size,projec_num]
        states_backward = states[1][-1]
        # states_h_concat:[batch_size,projec_num*2]
        states_h_concat = tf.concat(values=[states_forward.h, states_backward.h], axis=1, name="state_h_concat")
        # states_c_concat:[batch_size,cell_num*2]
        states_c_concat = tf.concat(values=[states_forward.c, states_backward.c], axis=1, name="state_c_concat")
        # 做成LSTMStateTuple
        encoder_states = tf.nn.rnn_cell.LSTMStateTuple(c=states_c_concat, h=states_h_concat)
        return encoder_outputs, encoder_states

    def LSTM_Simple_Encoder(self,inputs, cell_forward, cell_backward, seq_len, scope_name):
        '''
        单层LSTM做encoder,直接得到结果outputs和states
        outputs表示所有时间步的输出，对于双向网络，形式为
            前向结果：shape=(30, 80, 128)， 反向结果：shape=(30, 80, 128)
        states表示最后一步每一层的输出，形式为：
            前向：（LSTMStateTuple1,LSTMStateTuple2)，反向：（LSTMStateTuple1,LSTMStateTuple2)
        :param inputs: 输入
        :param cell_forward:前向LSTM cell
        :param cell_backward: 反向LSTM cell
        :param seq_len: 真实序列长度
        :param scope_name: scopr 名称
        :return: encoder之后的结果,所有时间的encoder_outputs和最终时刻的encoder_status
        '''
        # outputs,states=tf.nn.bidirectional_dynamic_rnn()
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_forward,
            cell_bw=cell_backward,
            inputs=inputs,
            sequence_length=seq_len,
            initial_state_fw=None,
            initial_state_bw=None,
            dtype=tf.float32,
            scope=scope_name
        )
        # print("outputs:",outputs)
        # print("states:",states)
        # 得到前向和反向outputs，并且拼起来
        outputs_forward = outputs[0]  # shape of h is [batch_size, max_time, projec_num]
        outputs_backward = outputs[1]  # shape of h is [batch_size, max_time, projec_num]
        encoder_outputs = tf.concat(values=[outputs_forward, outputs_backward],axis=2)  # [batch_size, max_time, projec_num*2]

        # 得到前向和反向states,因为两层以上，所以取最后一个[-1]
        states_forward = states[0]  # .c:[batch_size,num_units]   .h:[batch_size,projec_num]
        states_backward = states[1]
        # states_h_concat:[batch_size,projec_num*2]
        states_h_concat = tf.concat(values=[states_forward.h, states_backward.h], axis=1, name="state_h_concat")
        # states_c_concat:[batch_size,cell_num*2]
        states_c_concat = tf.concat(values=[states_forward.c, states_backward.c], axis=1, name="state_c_concat")
        # 做成LSTMStateTuple
        encoder_states = tf.nn.rnn_cell.LSTMStateTuple(c=states_c_concat, h=states_h_concat)
        return encoder_outputs, encoder_states

    def LSTM_Simple_Decoder(self,inputs,cell,initial_state,scope_name):
        '''
         单层LSTM做decoder,输出decoder之后的logits
        :param inputs: 输入
        :param cell:
        :param initial_state:
        :param scope_name:
        :return:
        '''
        outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=initial_state, scope=scope_name)
        return outputs              # outputs:[batch_size,time_steps,hidden_size*2]



class CnnUtil():
    '''
    CNN一些相关算法的的封装
    '''
    def __init__(self):
        pass

    def conv1d(self,inputs,
                    filters,
                    kernel_size,
                    strides=1,
                    dilation_rate=1,
                    casual=False,
                    activation=None,
                    kernel_initializer=None,
                    bias_initializer=tf.zeros_initializer(),
                    keep_prob=1.0,
                    regularizer=None,
                    name=None,
                    reuse=None):
        '''
        对于一个输入进行一维卷积并且进行dropout，是为tf.layers.conv1d的更高一层封装
        :param inputs: 输入数据,输入数据的形状为(batch,length,channels)
        :param filters: 卷积核数量,表现为输出特征图的channels数
        :param kernel_size: kernel尺寸，输入一个整数，表示1维卷积窗口的大小
        :param strides: 滑动步长，输入一个整数，表示卷积的步长
        :param dilation_rate: 卷积的空洞大小率
        :param casual: 是否是因果卷积
        :param activation: 激活函数
        :param kernel_initializer: 卷积核的初始化方式
        :param bias_initializer: bias的初始化方式
        :param keep_prob: drop_out保存的比率
        :param regularizer: regularizer
        :param name: 名字
        :param reuse: 是否reuse
        :return: 返回经过1d卷积并且dropout之后的结果，返回的形状为(batch,new_length,filters)
        '''
        #padding
        if casual:
            # padding(only padding left side,casual convlution)
            pad_left = dilation_rate * (kernel_size - 1)
            inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_left, 0], [0, 0]])
        else:
            pad_total = dilation_rate * (kernel_size - 1)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_left, pad_right], [0, 0]])
        #conv
        conv=tf.layers.conv1d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=name,
            reuse=reuse
        )
        #dropout
        # noise shape should be [N, 1, C]
        noise_shape = (tf.shape(conv)[0], tf.constant(1), tf.shape(conv)[2])
        out=tf.nn.dropout(x=conv,keep_prob=keep_prob,noise_shape=noise_shape)
        return out


    def conv1d_block(self,inputs,
                        filters,
                        kernel_size,
                        strides=1,
                        dilation_rate=1,
                        casual=False,
                        activation=None,
                        kernel_initializer=None,
                        bias_initializer=tf.zeros_initializer(),
                        keep_prob=1.0,
                        regularizer=None,
                        name="conv1d_block",
                        reuse=None):
        '''
        一维卷积残差块，因果卷积可选
        参考论文：
            《An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling》
        :param inputs: 输入数据,输入数据的形状为(batch,length,channels)
        :param filters: 卷积核数量,表现为输出特征图的channels数
        :param kernel_size: kernel尺寸，输入一个整数，表示1维卷积窗口的大小
        :param strides: 滑动步长，输入一个整数，表示卷积的步长
        :param dilation_rate: 卷积的空洞大小率
        :param casual: 是否是因果卷积
        :param activation: 激活函数
        :param kernel_initializer: 卷积核的初始化方式
        :param bias_initializer: bias的初始化方式
        :param keep_prob: drop_out保存的比率
        :param regularizer: regularizer
        :param name: 名字
        :param reuse: 是否reuse
        :return: 返回经过1d卷积并且dropout之后的结果，返回的形状为(batch,new_length,filters)
        '''
        #conv1
        conv1=self.conv1d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            casual=casual,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            keep_prob=keep_prob,
            regularizer=regularizer,
            name=name + "_conv1",
            reuse=reuse
        )
        print("shape of conv1:",conv1.shape)
        #conv2
        conv2 = self.conv1d(
            inputs=conv1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            casual=casual,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            keep_prob=keep_prob,
            regularizer=regularizer,
            name=name + "_conv2",
            reuse=reuse
        )
        print("shape of conv2:", conv2.shape)

        #residual,always use 1x1 conv
        res = self.conv1d(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=1,
            dilation_rate=1,
            casual=casual,
            activation=None,
            name=name+"_conv_1x1",
            reuse=reuse
        )
        outputs = tf.add(conv2, res)    # add residual
        print("output_shape:", outputs.shape)
        return outputs


    def weightNormConv1d(self):
        pass

    def weightNormConv1d_block(self):
        pass





    def separable_conv1d(self,inputs,filters,kernel_size,strides,dilation_rate=1,
                         activation=None,regularizer=None,keep_prob=0.5,name=None,reuse=None):
        '''
        对输入进行一维深度可分离卷积，并且进行dropout
        :param inputs:输入数据,输入数据的形状为(batch,length,channels)
        :param filters:卷积核数量,表现为输出特征图的channels数
        :param kernel_size:kernel尺寸，输入一个整数，表示1维卷积窗口的大小
        :param strides:滑动步长，输入一个整数，表示卷积的步长
        :param dilation_rate:空洞卷积率
        :param activation:激活函数
        :param regularizer:
        :param keep_prob:dropout保存的比率，要是不希望使用dropout，那么把keep_prob设置为1.0即可
        :param name:名字
        :param reuse:是否reuse
        :return:返回经过一维深度可分离卷积并且dropout之后的结果，返回的形状为(batch,new_length,filters)
        '''
        #padding
        pad_total = dilation_rate * (kernel_size - 1)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_left, pad_right], [0, 0]])
        #conv
        conv=tf.layers.separable_conv1d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="VALID",
            dilation_rate=dilation_rate,
            depth_multiplier=1,
            activation=activation,
            use_bias=True,
            depthwise_initializer=tf.initializers.variance_scaling(),
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_regularizer=regularizer,
            pointwise_regularizer=regularizer,
            bias_regularizer=regularizer,
            trainable=True,
            name=name,
            reuse=reuse
        )
        #dropout
        conv = tf.layers.dropout(inputs=conv, rate=1 - keep_prob)
        return conv

    def blocks_separable_conv1d(self,inputs,filters,kernel_size,strides,dilation_rate,activation,regularizer,keep_prob,level,reuse):
        '''
        一维可分离卷积残差块。
            参考论文：《An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling》
        :param inputs:输入数据,输入数据的形状为(batch,length,channels)
        :param filters:卷积核数量,表现为输出特征图的channels数
        :param kernel_size:kernel尺寸，输入一个整数，表示1维卷积窗口的大小
        :param strides:滑动步长，输入一个整数，表示卷积的步长
        :param dilation_rate:空洞卷积率
        :param activation:激活函数
        :param regularizer:
        :param keep_prob:dropout保存的比率
        :param level:层级，表示这是第几个blocks
        :param reuse:是否reuse
        :return:返回经过1d卷积块并且dropout之后的结果，返回的形状为(batch,new_length,filters)
        '''
        #conv layer 1
        conv_1=self.separable_conv1d(
            inputs=inputs,filters=filters,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,
            activation=activation,regularizer=regularizer,keep_prob=keep_prob,name="block"+str(level)+"conv_1",
            reuse=reuse,
        )
        # print("shape of conv_1:",conv_1.shape)
        #conv layer 2
        conv_2 = self.separable_conv1d(
            inputs=conv_1, filters=filters, kernel_size=kernel_size, strides=strides,dilation_rate=dilation_rate,
            activation=activation,regularizer=regularizer, keep_prob=keep_prob, name="block"+str(level)+"conv_2",
            reuse=reuse,
        )
        # print("shape of conv_2:", conv_2.shape)

        input_channels=inputs.get_shape().as_list()[-1]
        # print("input_channels:",input_channels)
        if input_channels==filters:
            res=inputs
        else:
            #1x1 conv
            res=self.separable_conv1d(
                inputs=inputs,filters=filters,kernel_size=1,strides=1,dilation_rate=1,activation=None,regularizer=None,
                keep_prob=1.0,name="block"+str(level)+"res",reuse=reuse
            )

        # print("shape of res:", res.shape)
        #add residual
        outputs=tf.add(conv_2,res)
        return outputs

    def casual_separable_conv1d(self, inputs, filters, kernel_size, strides, dilation_rate=1,
                         activation=None, regularizer=None, keep_prob=0.5, name=None, reuse=None):
        '''
        对输入进行一维深度可分离卷积，并且进行dropout
        :param inputs:输入数据,输入数据的形状为(batch,length,channels)
        :param filters:卷积核数量,表现为输出特征图的channels数
        :param kernel_size:kernel尺寸，输入一个整数，表示1维卷积窗口的大小
        :param strides:滑动步长，输入一个整数，表示卷积的步长
        :param dilation_rate:空洞卷积率
        :param activation:激活函数
        :param regularizer:
        :param keep_prob:dropout保存的比率，要是不希望使用dropout，那么把keep_prob设置为1.0即可
        :param name:名字
        :param reuse:是否reuse
        :return:返回经过一维深度可分离卷积并且dropout之后的结果，返回的形状为(batch,new_length,filters)
        '''
        # padding(only padding left side,casual convlution)
        pad_left = dilation_rate * (kernel_size - 1)
        inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_left, 0], [0, 0]])
        # conv
        conv = tf.layers.separable_conv1d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="VALID",
            dilation_rate=dilation_rate,
            depth_multiplier=1,
            activation=activation,
            use_bias=True,
            depthwise_initializer=tf.initializers.variance_scaling(),
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_regularizer=regularizer,
            pointwise_regularizer=regularizer,
            bias_regularizer=regularizer,
            trainable=True,
            name=name,
            reuse=reuse
        )
        # dropout
        conv = tf.layers.dropout(inputs=conv, rate=1 - keep_prob)
        return conv


    def conv2d(self):
        pass

    def separable_conv2d(self):
        pass

    def blocks_conv2d(self):
        pass



class AttentionUtil():
    '''
    常见注意力机制的一些基本封装，Attention是基础模块，这里不引用任何本文件的其他模块
    '''
    def __init__(self):
        self.basicUtil=BasicUtil()
        self.normalizeUtil=NormalizeUtil()

    def self_attention(inputs,keep_prob,name):
        '''
        对输入进行self_attention,实现参考论文《Attention Is All You Need》
        :param inputs: 输入，形状为(batch,length,channels)
        :param keep_prob: dropout保持的比率
        :return: 输出为input进行attention之后的tensor，形状为batch,length,channels)
        '''
        K_size = inputs.get_shape().as_list()[-1]
        V_size = inputs.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            #Key,shape(batch,length,channels)
            K=tf.layers.dense(
                inputs=inputs,
                units=K_size,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            K=tf.nn.dropout(x=K,keep_prob=keep_prob)            #dropout

            #Query,shape(batch,length,channels)
            Q = tf.layers.dense(
                inputs=inputs,
                units=K_size,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            Q = tf.nn.dropout(x=Q, keep_prob=keep_prob)  # dropout

            #Value,shape(batch,length,channels)
            V = tf.layers.dense(
                inputs=inputs,
                units=V_size,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            V = tf.nn.dropout(x=V, keep_prob=keep_prob)  # dropout

            #矩阵计算，self attention的核心步骤，这里注意一下三维矩阵的矩阵运算规则（batch那一维不参与运算）
            logits=tf.matmul(a=Q,b=K,transpose_b=True)              #需要将K进行转置
            logits=logits/tf.sqrt(K_size)                           #缩放因子
            weights = tf.nn.softmax(logits, name="attention_weights")
            output = tf.matmul(weights, V)
            # print("outout.shape",output.shape)
        return output


    def scaled_dot_product_attention(self,Q,K,V,keep_prob,name):
        '''
        有缩放因子的self attention，实现参考论文《Attention Is All You Need》 3.2.1
        :param Q:Query,形状为(batch,length,channels)
        :param K:Key，形状为(batch,length,channels)
        :param V:Value，形状为(batch,length,channels)
        :param keep_prob:dropout阶段的保留比率
        :param name:名字
        :return:进行attention之后的结果,形状和V的形状一样为(batch,length,channels)
        '''
        with tf.variable_scope(name):
            d_k=Q.get_shape().as_list()[-1] #channels size(dimension)
            #矩阵计算，核心步骤，这里要注意：三维矩阵的矩阵运算规则（batch那一维不参与运算）
            logits=tf.matmul(a=Q,b=K,transpose_b=True)
            logits/=d_k**0.5        #scale
            weights=tf.nn.softmax(logits=logits)
            outputs=tf.matmul(a=weights,b=V)
            #dropout
            outputs = tf.layers.dropout(inputs=outputs, rate=1 - keep_prob)
        return outputs


    def multi_head_attention(self,queries,keys,values,num_heads,keep_prob,name):
        '''
        多头Attention，实现参考《Attention Is All You Need》 3.2.2
        :param queries: Query,形状为(batch,length,channels)
        :param keys: Key，形状为(batch,length,channels)
        :param values: Value，形状为(batch,length,channels)
        :param num_heads: 头的数量
        :param keep_prob: dropout阶段的保持比率
        :param name: scope名
        :return: 进行了multi head attention之后的结果
        '''
        with tf.variable_scope(name):
            d_model=queries.get_shape().as_list()[-1]   #channels size(dimension)
            #线性变换，注意这里激活函数一定要是None
            Q = tf.layers.dense(
                inputs=queries,
                units=d_model,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            #Q = tf.nn.dropout(x=Q, keep_prob=keep_prob)  # dropout

            K = tf.layers.dense(
                inputs=keys,
                units=d_model,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            #K = tf.nn.dropout(x=K, keep_prob=keep_prob)  # dropout

            V = tf.layers.dense(
                inputs=values,
                units=d_model,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )
            #V = tf.nn.dropout(x=V, keep_prob=keep_prob)  # dropout
            #split according to channels
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (batch*num_heads,length,d_model/num_heads)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (batch*num_heads,length,d_model/num_heads)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (batch*num_heads,length,d_model/num_heads)

            #self attention[batch*num_heads,length,d_model/num_heads]
            outputs=self.scaled_dot_product_attention(Q=Q_,K=K_,V=V_,keep_prob=keep_prob,name="self_attention")

            #restore shape
            outputs=tf.concat(tf.split(outputs,num_heads,axis=0),axis=2)    #(batch,length,channels)
        return outputs


    def transformer_block(self,inputs,num_heads,units,keep_prob,name):
        '''
        transformer块实现，实现参考《Attention Is All You Need》 Figure 1
        :param inputs: 输入，形状为(batch,length,channels)
        :param num_heads: 头的数量
        :param units: feed forward 层第一层结点数量
        :param keep_prob: dropout保留的比率
        :param name: scope名
        :return:
        '''
        with tf.variable_scope(name):
            #Multi-head Attention Process
            outputs_1=self.multi_head_attention(
                queries=inputs,
                keys=inputs,
                values=inputs,
                num_heads=num_heads,
                keep_prob=keep_prob,
                name="multi_head_attention"
            )
            outputs_1+=inputs   #residual connection
            outputs_1=self.normalizeUtil.layer_norm(outputs_1,name="layer_norm_1")   #layer norm

            #feed forwad Process
            outputs_2=self.basicUtil.feed_forward(inputs=outputs_1,units=units,name="feed_forward")
            outputs_2+=outputs_1    #residual connection
            outputs_2=self.normalizeUtil.layer_norm(outputs_2,name="layer_norm_2")  #layer_norm
        return outputs_2


class NormalizeUtil():
    '''
    normalize的一些基本封装
    '''
    def __init__(self):
        pass

    def batch_norm(self):
        pass

    def weight_norm(self):
        pass

    def layer_norm(self,inputs,epsilon=1e-8,name="layer_norm"):
        '''
        layer norm,实现参考《》
        :param inputs:输入，维度2到更多，其中第一维是batch_size
        :param epsilon: 非常小的浮点数，用来防止除0的错误
        :param name:scope名字
        :return: input经过layer norm之后的形式
        '''
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
        return outputs




if __name__=="__main__":
    #input=tf.ones(shape=(2,3,4),dtype=tf.float32,name="input")
    input=tf.constant(value=[
        [
            [1, 2, 1, 1, 1, 1, 2],
            [1, 2, 1, 1, 2, 3, 4],
            [1, 1, 2, 2, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [1, 2, 1, 1, 3, 2, 1],
            [1, 2, 1, 1, 1, 2, 3],
            [1, 1, 2, 2, 3, 2, 1],
            [1, 1, 1, 1, 1, 2, 3],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
    ],dtype=tf.float32,name="input")
    obj=CnnUtil()
    obj.conv1d_block(inputs=input,filters=4,kernel_size=3,strides=1,casual=False)

