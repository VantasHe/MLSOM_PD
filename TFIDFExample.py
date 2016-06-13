import os
import sys
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer()
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
CountArray = X.toarray()
print(CountArray)
print(vectorizer.get_feature_names())

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(CountArray)
print(tfidf.toarray())

def Tfidf(filelist) :
　　path = './segfile／'
    corpus = []  # Store the result of the doucment feature
    for ff in filelist :
        fname = path + ff
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content)    

    vectorizer = CountVectorizer()    
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    
    word = vectorizer.get_feature_names() # Get keywords from all the corpus
    weight = tfidf.toarray()              # Imply the Matrix of Tf-idf
    
    sFilePath = './tfidffile'
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)

    # Store Tf-idf results of whole documet to './tfidffile'
    for i in range(len(weight)) :
　　　　 print u"--------Writing all the tf-idf in the",i,u" file into ",sFilePath+'/'+string.zfill(i,5)+'.txt',"--------"
        f = open(sFilePath+'/'+string.zfill(i,5)+'.txt','w+')
        for j in range(len(word)) :
            f.write(word[j]+"    "+str(weight[i][j])+"\n")
        f.close()