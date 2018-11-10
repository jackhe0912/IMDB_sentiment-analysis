import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import word2vec,Word2Vec
import nltk.data
import logging

from sklearn.ensemble import RandomForestClassifier

#词向量模型参数设置

num_features = 300      # 词向量维度
min_word_count = 40     # 最低字数
num_workers = 4         # 并行的线程个数
context = 10            # 上下文窗口大小
downsampling = 1e-3     # 频词下采样设置


def loadset():
    """
    加载数据，并将有标记的训练数据和没标记的训练数据合并，Word2vec可以通过无标记的数据来进行学习
    :return:
    """
    label_train=pd.read_csv('D:/machine learning/kaggle/Movies/all/labeledTrainData.tsv',
                  header=0,delimiter='\t',quoting=3)
    unlabel_train=pd.read_csv('D:/machine learning/kaggle/Movies/all/unlabeledTrainData.tsv',
                  header=0,delimiter='\t',quoting=3)
    test_data=pd.read_csv('D:/machine learning/kaggle/Movies/all/testData.tsv',
                  header=0,delimiter='\t',quoting=3)

    train_data=pd.concat([label_train,unlabel_train],ignore_index=True)

    return train_data,label_train,test_data


def review2words(raw_review, remove_stopwords=False):
    """
    数据清洗，进行word2vec时不去停用词，词越丰富越会产生更高质量的词向量
    :param raw_review:
    :param remove_stopwords:
    :return:
    """
    # 去除html标签
    review_data = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 去掉非字母的符号
    review_data = re.sub('[^a-zA-Z]', ' ', review_data)
    # 小写化所有词，并转换成词list
    review_list = review_data.lower().split()
    # 选择性去除停用词
    if remove_stopwords:
        stopwords_set = set(stopwords.words('english'))
        review_list = [w for w in review_list if not w in stopwords_set]
    return review_list

def review2sentences(review, tokenizer, remove_stopwords=False):
    """
    将段落分成句子，再将句子 切分成单词list
    :param review:
    :param tokenizer:
    :param remove_stopwords:
    :return:
    """
    #将段落分成句子
    raw_sentences = tokenizer.tokenize(review.strip())
    #将每个句子再切分成单词list
    sentences = []
    for each_sentence in raw_sentences:
        if len(each_sentence) > 0:
            sentences.append(review2words(each_sentence, remove_stopwords))
    return sentences

def trainmodel():
    """
    训练word2vec模型
    :return:
    """
    train_data,_,_=loadset()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []
    for review in train_data['review']:
        sentences += review2sentences(review, tokenizer)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = word2vec.Word2Vec(sentences,
            workers=num_workers,
            size=num_features,
            min_count = min_word_count,
            window = context,
            sample = downsampling)

    model.init_sims(replace=True)
    model_name = '300features_40minwords_10context'
    model.save(model_name)

def makeFeatureVec(words, model, num_features):
    """
    给定的段落中的词向量平均
    :param words:
    :param model:
    :param num_features:
    :return:
    """
    feature_vec=np.zeros((num_features),dtype='float32')
    nwords = 0
    index2word_set = set(model.wv.index2word)  #词向量模型中的词名称

    for word in words:
        if word in index2word_set:
            nwords +=1
            feature_vec=np.add(feature_vec,model[word])

    feature_vec=np.divide(feature_vec,nwords)
    return feature_vec

def getAvgFeatureVecs(reviews, model, num_features):
    """
    对数据集所有的评论进行特征向量平均，并存在矩阵里
    :param reviews:
    :param model:
    :param num_features:
    :return:
    """
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter +=1
    return reviewFeatureVecs

def main():

    _,train_data,test_data=loadset()
    model=Word2Vec.load('300features_40minwords_10context')

    clean_train_data=[]
    for review in train_data['review']:
        clean_train_data.append(review2words(review,remove_stopwords=True))

    train_data_vec=getAvgFeatureVecs(clean_train_data,model,num_features)

    clean_test_data=[]
    for review in test_data['review']:
        clean_test_data.append(review2words(review,remove_stopwords=True))
    test_data_vec=getAvgFeatureVecs(clean_test_data,model,num_features)

    forest = RandomForestClassifier(n_estimators=150,max_features='sqrt',max_depth=8,min_samples_split=4)
    forest.fit(train_data_vec,train_data['sentiment'])

    result=forest.predict(test_data_vec)

    output = pd.DataFrame(data={'id': test_data['id'], 'sentiment': result})
    output.to_csv('result_Word2Vec_AverageVectors.csv', index=False, quoting=3)

trainmodel()
main()




