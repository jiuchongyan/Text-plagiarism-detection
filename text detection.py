__author__ = 'samsung'
import pandas as pd
import numpy as np
import jieba

#数据加载
news = pd.read_csv('sqlResult.csv',encoding='gb18030')
print(news.shape)
print(news.head())

#判断content是否有空值
news[news.content.isna()].head()
#去掉content为空的行
news.dropna(subset=['content'],inplace=True)
print(news.shape)
#加载停用词
with open('chinese_stopwords.txt','r',encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]
print(stopwords)

#分词
def split_text(text):
    text = text.replace(' ','').replace('\n','')
    text2 = jieba.cut(text)
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result
split_text(news.iloc[0].content)

#对每一个content进行分词)
corpus = list(map(split_text, [str(i) for i in news.content]))
corpus

#提取文本特征，计算corpus的TF－IDF
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
countvectorizer = CountVectorizer(encoding='gb18030',min_df=0.015) #计算词频
tfidftransformer = TfidfTransformer() #计算tf-idf
countvector = countvectorizer.fit_transform(corpus)  #统计出词频
tfidf = tfidftransformer.fit_transform(countvector)
print(tfidf.shape)
#标记是否为新华社自己的新闻，用source标记
label = list(map(lambda source:1 if '新华社' in str(source) else 0,news.source))

#数据集切分
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(tfidf.toarray(),label,test_size=0.3)

#用贝叶斯来预测分类
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
model = MultinomialNB()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print('准确率',accuracy_score(y_predict,y_test))
print('精确率',precision_score(y_predict,y_test))
print('召回率',recall_score(y_predict,y_test))

#使用模型进行风格预测
prediction = model.predict(tfidf.toarray())  #预测风格
labels = np.array(label) #真实风格
#将上两列拼成dataframe
compare_news_index = pd.DataFrame({'prediction':prediction,'label':labels})
#识别出怀疑的抄袭对象，prediction=1，label=0
copy_news_index = compare_news_index[(compare_news_index.prediction==1)&(compare_news_index.label==0)].index

#实际为新华社新闻的index
xhs_news_index = compare_news_index[compare_news_index.label==1].index
print(len(xhs_news_index))
#可能为抄袭的新闻数
print(len(copy_news_index))