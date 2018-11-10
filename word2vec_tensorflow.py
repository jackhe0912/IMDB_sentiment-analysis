
from bs4 import BeautifulSoup
import re
import collections
from collections import Counter
from nltk.corpus import stopwords
import random
import math
import pandas as pd
import numpy as np
import tensorflow as tf

class SkipGram:
    def __init__(self):
        self.min_count=5   #保留模型中的词表的最低词频
        self.data_index=0
        self.batch_size = 200
        self.window_size=5      #skipgram时窗口大小
        self.embedding_size=300  #嵌入向量维数
        self.n_sampled=100    #负采样样本数
        self.num_step = 100000    # 定义最大迭代次数

    def load_dataset(self):
        """
        加载数据
        :return:
        """
        label_train = pd.read_csv('D:/machine learning/kaggle/Movies/all/labeledTrainData.tsv',
                                  header=0, delimiter='\t', quoting=3)
        unlabel_train = pd.read_csv('D:/machine learning/kaggle/Movies/all/unlabeledTrainData.tsv',
                                    header=0, delimiter='\t', quoting=3)

        return label_train, unlabel_train

    def review_to_words(self,raw_review):
        """
        去除评论中的标签，标点，停用词，并小写化
        :param raw_review:
        :return:
        """
        # 去除html标签
        review_data = BeautifulSoup(raw_review, 'html.parser').get_text()
        # 去掉标点符号
        review_data = re.sub('[^a-zA-Z]', ' ', review_data)
        # 小写化所有词，并转黄成词list
        review_list = review_data.lower().split()
        # 引进停用词
        stopwords_set = set(stopwords.words('english'))
        # 去除review_list中的停用词
        useful_review = [w for w in review_list if not w in stopwords_set]

        return useful_review

    def get_words(self):
        """
        从所有评论中得到词的集合
        :return:
        """
        label_train, unlabel_train = self.load_dataset()
        train = pd.concat([label_train, unlabel_train], ignore_index=True)
        words = []
        for i in range(0, len(train.review)):
            words.extend(self.review_to_words(train.review[i]))
        return words

    def build_new_data(self,words):
        """
        创建词汇表，过滤低频词，其余词标记为unk，并将样本词汇映射成索引数字
        :param words:
        :return:
        """
        count=[['unk',-1]]
        count.extend([item for item in Counter(words).most_common() if item[1]>=self.min_count])
        dictionary=dict()
        for word,_ in count:
            dictionary[word]=len(dictionary)
        data=[]          #对原words列表中的单词使用字典中的ID进行编号，即将单词转换成整数，储存在data列表中，同时对UNK进行计数
        unk_count=0
        for word in words:
            if word in dictionary:
                index=dictionary[word]
            else:
                index=0
                unk_count+=1
            data.append(index)
        count[0][1]=unk_count
        # 将dictionary中的key与value反转，即可以通过ID找到对应的单词，保存在reversed_dictionary中
        reversed_dictionary=dict(zip(dictionary.values(),dictionary.keys()))

        return data,count,dictionary,reversed_dictionary

    def generate_batch(self,batch_size,window_size,data):
        """

        :param batch_size: 每批次训练的样本数
        #:param num_skips:  为每个单词生成多少样本，batch_size必须是num_skips的整数倍
        :param window_size:   单词最远可联系的距离  2*skip_window>=num_skips
        :param data:
        :return:
        """
        #assert batch_size % num_skips == 0
        #assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)   #建一个batch大小的一维数组，保存任意单词
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)   #建立一个（batch，1)大小的二维数组
        span = 2 * window_size + 1              #  窗口的大小,结构为[ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)  # 建立一个结构为双向队列的缓冲区，大小不超过3，实际上是为了构造batth以及labels，采用队列的思想
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index=(self.data_index + 1) % len(data)
        for i in range(batch_size//(2*window_size)):              #batch_size一定是Num_skips的倍数，保证每个batch_size都能够用完num_skips
            target = window_size
            targets_to_avoid =[window_size]
            for j in range(2*window_size):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * 2*window_size + j] = buffer[window_size]
                labels[i * 2*window_size + j, 0] = buffer[target]
            buffer.append(data[self.data_index])
            self.data_index=(self.data_index+1)%len(data)

        return batch,labels

    def train_wordvec(self,vocabulary_size, batch_size, embedding_size, window_size, n_sampled,data):
        """

        :param vocabulary_size:
        :param batch_size:
        :param embedding_size:
        :param window_size:
        :param n_sampled:负采样时，标签为负的样本数
        :param data:
        :return:
        """
        graph = tf.Graph()
        with graph.as_default():

            train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
            train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])

            # 随机初始化一个值于介于-1和1之间的随机数，矩阵大小为词表大小乘以词向量维度
            embedding=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
            # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。用于查找对应的wordembedding，将输入序列向量化
            embed=tf.nn.embedding_lookup(embedding,train_inputs)

            softmax_weights=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0 /math.sqrt(embedding_size)))
            softmax_bias=tf.Variable(tf.zeros([vocabulary_size]))

            loss=tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights,softmax_bias,train_labels,embed,n_sampled,vocabulary_size)) #负采样损失

            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))  #计算每个词向量的模，并进行单位归一化，保留词向量维度
            normalized_embeddings = embedding / norm



        with tf.Session(graph=graph) as sess:

            sess.run(tf.global_variables_initializer())   #初始化运行
            average_loss = 0

            for step in range(1,self.num_step):

                batch, labels=self.generate_batch(batch_size,window_size,data)
                # feed_dict是一个字典，在字典中需要给出每一个用到的占位符的取值
                feed_dict={train_inputs:batch,train_labels:labels}

                _,train_loss=sess.run([optimizer,loss],feed_dict=feed_dict)
                average_loss+=train_loss

                if step % 1000 == 0:
                    if step>0:
                        average_loss /=1000
                    print('average_loss at iteration',step,':',average_loss)
                    average_loss=0

            final_embeddings =sess.run(normalized_embeddings)
            np.save('embeding.npy',final_embeddings)

    def train(self):
        """
        训练主函数
        :return:
        """

        words=self.get_words()
        data, count, dictionary, reversed_dictionary=self.build_new_data(words)
        np.save('wordlist.npy',reversed_dictionary)
        vocabulary_size=len(count)
        #batch, labels=self.generate_batch(self.batch_size,self.window_size,vocab)
        self.train_wordvec(vocabulary_size,self.batch_size,self.embedding_size,self.window_size,self.n_sampled,data)


def test():

    vector = SkipGram()
    vector.train()

test()







