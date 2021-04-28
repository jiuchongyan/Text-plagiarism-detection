__author__ = 'samsung'
import pandas as pd
import numpy as np
import jieba

#���ݼ���
news = pd.read_csv('sqlResult.csv',encoding='gb18030')
print(news.shape)
print(news.head())

#�ж�content�Ƿ��п�ֵ
news[news.content.isna()].head()
#ȥ��contentΪ�յ���
news.dropna(subset=['content'],inplace=True)
print(news.shape)
#����ͣ�ô�
with open('chinese_stopwords.txt','r',encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]
print(stopwords)

#�ִ�
def split_text(text):
    text = text.replace(' ','').replace('\n','')
    text2 = jieba.cut(text)
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result
split_text(news.iloc[0].content)

#��ÿһ��content���зִ�)
corpus = list(map(split_text, [str(i) for i in news.content]))
corpus

#��ȡ�ı�����������corpus��TF��IDF
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
countvectorizer = CountVectorizer(encoding='gb18030',min_df=0.015) #�����Ƶ
tfidftransformer = TfidfTransformer() #����tf-idf
countvector = countvectorizer.fit_transform(corpus)  #ͳ�Ƴ���Ƶ
tfidf = tfidftransformer.fit_transform(countvector)
print(tfidf.shape)
#����Ƿ�Ϊ�»����Լ������ţ���source���
label = list(map(lambda source:1 if '�»���' in str(source) else 0,news.source))

#���ݼ��з�
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(tfidf.toarray(),label,test_size=0.3)

#�ñ�Ҷ˹��Ԥ�����
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
model = MultinomialNB()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print('׼ȷ��',accuracy_score(y_predict,y_test))
print('��ȷ��',precision_score(y_predict,y_test))
print('�ٻ���',recall_score(y_predict,y_test))

#ʹ��ģ�ͽ��з��Ԥ��
prediction = model.predict(tfidf.toarray())  #Ԥ����
labels = np.array(label) #��ʵ���
#��������ƴ��dataframe
compare_news_index = pd.DataFrame({'prediction':prediction,'label':labels})
#ʶ������ɵĳ�Ϯ����prediction=1��label=0
copy_news_index = compare_news_index[(compare_news_index.prediction==1)&(compare_news_index.label==0)].index

#ʵ��Ϊ�»������ŵ�index
xhs_news_index = compare_news_index[compare_news_index.label==1].index
print(len(xhs_news_index))
#����Ϊ��Ϯ��������
print(len(copy_news_index))