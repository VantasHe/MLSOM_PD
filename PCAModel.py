import os
import sys
import string
import time

import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import somoclu
import matplotlib.pyplot as plt

def PCA_Model(X_std):
    sklearn_pca = sklearnPCA(n_components=200)
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    return Y_sklearn
    
def transToHistogram(rootOfDataBase, method = 'Mean'):
    path = rootOfDataBase + 'outputData/'   # Load index of directory of preprocessing datasets.
    filelist = []
    files = os.listdir(path)    # Get list of directory.
    for f in files :
        if (f[0] == '.') :      # Exclude the system file .
            pass                # Do nothing
        else :
            filelist.append(f)  # Append file list

    """ Load all corpus"""
    corpus = []  # Store the result of the document feature
    for ff in filelist :
        fname = path + ff
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content)
    
    """ Calculate tf(term frequency)"""
    term_fq_vectorizer = CountVectorizer()      # Instance to count all terms in each document 
    term_fq = np.array(term_fq_vectorizer.fit_transform(corpus).toarray())  # Transform tf into array
    np.savetxt(rootOfDataBase+"term_frequency.csv", term_fq, delimiter=',')    #save as csv
    if (method == 'Mean'):
        term_fq_count = term_fq.sum(axis=1).reshape((term_fq.shape[0],1))     # Sum the times of terms in a document.
    elif (method == 'Max'):
        term_fq_count = np.amax(term_fq, axis=1).reshape((term_fq.shape[0],1))     # Find Maximun in a row.
    else:
        exit(0)
    """ Expand total term frequency array to Matrix """
    totalTermForEachFile = np.array(term_fq_count) 
    for i in range(term_fq.shape[1]-1) : 
        totalTermForEachFile = np.concatenate((totalTermForEachFile, term_fq_count), axis = 1)
        
    histogram = term_fq / totalTermForEachFile
    
    if (method == 'Mean'):
        np.savetxt(rootOfDataBase+"histogramMean.csv", histogram, delimiter=',')    #save as csv
    elif (method == 'Max'):
        np.savetxt(rootOfDataBase+"histogramMax.csv", histogram, delimiter=',')    #save as csv
    else:
        exit(0)
    
    
    return histogram, filelist
    
if __name__ == "__main__":
    #print(histogramByMean())
    rootOfDataBase = '../Training_Data/'
    msz0 = 10
    msz1 = 10
    data, doc = transToHistogram(rootOfDataBase, method = 'Mean')
    new_data = np.float32(data)
    
    print("SOM Trainin & clustering...", end="")
    TimeStart = time.time()
    som = somoclu.Somoclu(msz0, msz1, data=new_data, maptype="toroid")
    som.train()
    som.cluster()
    TimeEnd = time.time()
    print("[done] : {0:f} sec".format(TimeEnd - TimeStart))
    
    codebook = som.codebook
    label = doc
    bmus = som.bmus
    cluseter = som.clusters
    matchList = dict(zip(label,bmus))
    nCluster = dict()
    for key, val in matchList.items() :
        print(str(key), ">>", val, ":", cluseter[val[0]][val[1]])
        clName = str(cluseter[val[0]][val[1]])
        if clName in nCluster :
            nCluster[clName].append(str(key))
        else :
            nCluster[clName] = [key]
    fo = open("../Training_Data/result.txt", "w+")
    for cl, name in nCluster.items() :        
        fo.write(str(cl) + ' ' + str(name))
        fo.write('\n')
    fo.close()
    
    som.view_umatrix(bestmatches=True, labels=label)