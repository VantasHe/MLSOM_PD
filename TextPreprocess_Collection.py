import os
import sys
import string
import time

import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import LookupIndex

cachedStopWords = stopwords.words("english")
    
def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)
    
""" Method of preprocessing file , like stemming. """
def preprocessFile(argv, path) :
    sFilePath = '../Training_Data/segfile'   # Create the directory to store segments of files.
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)

    filename = argv                 
    f = open(path+filename,'r+')    # Open file, mode: read only.
    file_list = f.read()            # Read file content and store as string.
    f.close()                       # Close file.
    
    """ Remove punctuation from document """
    deletetable = str.maketrans({key:" " for key in string.punctuation})    # Make translation table of punctuation.
    text = file_list.translate(deletetable)     # Remove punctuation from file.
    #text = ' '.join([word for word in file_list.split() if word not in cachedStopWords])
    """ Turn document into lowercase and remove Stopwords """
    text = ' '.join([word for word in text.lower().split() if word not in cachedStopWords])
    seg_list = text.split() 
    #seg_list = file_list.split()
    
    """ Method : stemming """
    st = LancasterStemmer()
    result = []
    for seg in seg_list :
        st_seg = st.stem(seg)
        if(st_seg != '' and st_seg != '\n'):
            result.append(st_seg)            
    """
    wnl= WordNetLemmatizer()
    for seg in seg_list :
        wnl_seg = wnl.lemmatize(seg)
        if(wnl_seg != '' and wnl_seg != '\n'):
            result.append(wnl_seg)
    """

    """ Write into File """
    f = open(sFilePath+"/"+filename+"-seg.txt","w+")
    f.write(' '.join(result))
    f.close()

""" Calculate Tf-idf and the weight of each term """
def Tfidf(filelist) :
    path = '../Training_Data/segfile/'   # Load index of directory of preprocessing datasets.
    filelist = []
    files = os.listdir(path)    # Get list of directory.
    for f in files :
        if (f[0] == '.') :      # Exclude the root '.' .
            pass                # Do nothing
        else :
            filelist.append(f)  # Append file list

    """ Load all corpus"""
    corpus = []  # Store the result of the doucment feature
    for ff in filelist :
        fname = path + ff
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content)

    """
    vectorizer = CountVectorizer()    
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names() # Get keywords from all the corpus
    weight = tfidf.toarray()              # Imply the Matrix of Tf-idf
    """

    """ Calculate tf(term frequency) and df(document term)"""
    term_fq_vectorizer = CountVectorizer()      # Instance to count all terms in each document 
    document_fq_vectorizer = CountVectorizer(binary=True)   # Instance to count the term whether in document
    
    term_fq = np.array(term_fq_vectorizer.fit_transform(corpus).toarray())  # Transform tf into array
    document_fq = np.array(document_fq_vectorizer.fit_transform(corpus).toarray())  # Transform df into array
    
    term_fq_count = term_fq.sum(axis=0)     # Sum the times of term in all document.
    document_fq_count = document_fq.sum(axis=0) # Sum the times of term appeared in document.
   
    word = term_fq_vectorizer.get_feature_names() # Get keywords from all the corpus
    count_document = document_fq.shape[0]   # Get the number of document.
    idf_temp = np.ones(document_fq.shape[1]) * count_document   # Create constant array
    idf = np.log2(idf_temp / document_fq_count) # Get the value of idf
    weight =  idf * np.sqrt(term_fq_count)  # Calculate weight of each term
    
    """ Sort terms by weight """
    dictionary = dict(zip(word, weight))     # Bind term name with its weight and transform to dictionary.
    sort_dict = sorted(dictionary.items(), key=lambda d: d[1], reverse=True)    # Sort descending by value. 
    
    """ Write to file """
    sFilePath = '../Training_Data/tfidffile'
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)
        
    """
    # Store Tf-idf results of whole documet to './tfidffile'
    for i in range(len(weight)) :
        print("--------Writing all the tf-idf in the", i, " file into ", sFilePath + '/' + str(i).zfill(5) + '.txt', "--------")
        f = open(sFilePath+'/'+ str(i).zfill(5) +'.txt','w+')
        for j in range(len(word)) :
            f.write(word[j]+"    "+str(weight[i][j])+"\n")
        f.close()
    """

    print("--------Writing all the tf-idf in the Database file into ", sFilePath + '/' + 'TotalBase.txt', "--------")
    f = open(sFilePath+'/' + 'TotalBase.txt', 'w+')
    for pair in sort_dict :
        f.write( str(pair[0]) + "\t" + str(pair[1]) + "\n" )
    f.close()
        
if __name__ == "__main__":
    argv = "/Users/vick/Documents/Training_Data/trainingfolder/"
    (allfile, path) = LookupIndex.lookupFileList(argv)
    
    timeStart = time.time()
    for ff in allfile :
        print("Using stem on " + ff)
        preprocessFile(ff,path)
    timeEnd = time.time()
    print("Preprocessing time : {0:f} sec".format(timeEnd - timeStart))
    
    timeStart = time.time()
    Tfidf(allfile)
    timeEnd = time.time()
    print("Tf-idf time : {0:f} sec".format(timeEnd - timeStart))