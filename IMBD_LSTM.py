
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from random import randint

class IMDB_LSTM:
    def __init__(self):
        self.max_seq_num=250
        self.num_review = 25000
        self.batch_size=24
        self.num_labels=2
        self.lstm_units=64
        self.iteration=100000


    def load_set(self):
        """
        加载训练集和测试集
        :return:
        """
        train=pd.read_csv('labeledTrainData.tsv',header=0, delimiter='\t', quoting=3)
        #test=pd.read_csv('testData.tsv')
        sentiment_labels=[]
        for i in train['sentiment']:
            sentiment_labels.append(i)

        return train,sentiment_labels

    def load_dict_embedding(self):
        """
        加载词典和词向量矩阵
        :return:
        """
        words_dict=np.load('wordlist.npy')     #载入word列表

        words_dict=words_dict.tolist()          #将array转换成list

        words_list = list(words_dict.values())  # 将word_dict中的值存为list

        word_vectors=np.load('embeding.npy')   #载入文本向量

        return word_vectors,words_list

    def cleaned_review(self,raw_review):
        """

        :param raw_review:
        :return:
        """
        review_data = raw_review.replace('<br />',' ')   # 去除html标签

        review_data = re.sub('[^a-zA-Z]', ' ', review_data)    # 去掉标点符号

        return review_data.lower().split()        # 小写化所有词，并转换成词list,然后返回

    def get_input_matrix(self,reviews,words_list):
        """
        把评论转换成输入25000*250矩阵，如果评论长度小于250，小于的部分的特征值设置为0，如果评论中的词不在词汇表，也标记为0
        :param reviews: 未经处理过的评论
        :param words_dict:
        :return:
        """

        input_matrix=np.zeros((self.num_review,self.max_seq_num),dtype='int32') #定义输入矩阵
        for i in range(0,len(reviews)):
            clean_review=self.cleaned_review(reviews[i])
            index_count=0
            for word in clean_review:
                try:
                    input_matrix[i][index_count]=words_list.index(word)
                except ValueError:
                    input_matrix[i][index_count]=words_list.index('unk')
                index_count+=1
                if index_count>=self.max_seq_num:
                    break
        np.save('input_matrix.npy', input_matrix)
        return input_matrix

    def get_batch_data(self,input_matrix,sentiment_labels):
        """

        :param input_matrix:
        :param sentiment_label:
        :return:
        """
        labels=[]
        arr=np.zeros([self.batch_size,self.max_seq_num])
        for i in range(self.batch_size):
            num=randint(0,24999)
            if sentiment_labels[num]==1:
                labels.append([1,0])
            else:
                labels.append([0,1])
            arr[i]=input_matrix[num-1:num]
        return arr,labels

    def train_lstm(self,word_vectors,arr,data_labels):

        tf.reset_default_graph()  #利用这个可清空defualt graph以及nodes

        labels=tf.placeholder(tf.float32,[self.batch_size,self.num_labels])
        input_data=tf.placeholder(tf.int32,[self.batch_size,self.max_seq_num])


        inputs=tf.nn.embedding_lookup(word_vectors,input_data)

        lstmCell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_units)
        lstmCell = tf.nn.rnn_cell.DropoutWrapper(lstmCell, output_keep_prob=0.75)
        _,states=tf.nn.dynamic_rnn(lstmCell,inputs,dtype=tf.float32)    #tf.nn.dynamic_rnn的作用是展开整个网络，并构建一整个RNN模型

        weight=tf.Variable(tf.truncated_normal([self.lstm_units,self.num_labels]))
        bias=tf.Variable(tf.constant(0.1,shape=[self.num_labels]))
        #outputs=tf.transpose(outputs,[1,0,2]) #对outputs进行转置
        #last=tf.gather(outputs,int(outputs.get_shape()[0])-1)  #得到最后状态的输出
        prediction=(tf.matmul(states.h,weight)+bias)

        #correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))   #转换成bool值
        #accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))            #评估函数，将bool值转换成float

        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=labels))
        optimizer=tf.train.AdamOptimizer().minimize(loss)

        sess=tf.InteractiveSession()
        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(self.iteration):
            nextBatch,nextBatchLabels=arr,data_labels
            sess.run(optimizer,{input_data:nextBatch,labels:nextBatchLabels})
            if (i%10000==0 and i!=0):
                save_path=saver.save(sess,'pretrained_lstm.ckpt',global_step=i)
                print('saved to %s' % save_path)

    def main(self):

        train,sentiment_labels=self.load_set()
        word_vectors, words_list=self.load_dict_embedding()
        input_matrix=self.get_input_matrix(train.review,words_list)
        arr,labels=self.get_batch_data(input_matrix, sentiment_labels)
        self.train_lstm(word_vectors,arr,labels)


def test():

    vector = IMDB_LSTM()
    vector.main()

test()




















