import os
import sys
import string
import time

import LookupIndex

def CreateVocabularyBase(pathOfTfidf, filename):
    firstT = 700
    dictionary = {}
    with open(pathOfTfidf+filename, 'r') as file:
        for line in file:
            (key, val) = line.split()
            dictionary[str(key)] = val
    sort_dict = sorted(dictionary.items(), key=lambda d: d[1], reverse=True)    # Sort descending by value.
    
    f = open(pathOfTfidf + 'VocabularyBase.txt', 'w+')
    for pair in sort_dict[:firstT] :
        f.write(str(pair[0]) + ", " )
    f.close()
    return dictionary
    
def filterByVocab(pathOfSeg, filename, pathOfVocab):
    with open(pathOfVocab, 'r') as fileVocab:
        for line in fileVocab:
            vocabBase = line.split(', ')
    fi = open(pathOfSeg+filename, 'r')
    fi_context = fi.read()
    text = ' '.join([word for word in fi_context.split() if word in vocabBase])
    
    sFilePath = '../Training_Data/outputData/'   # Create the directory to store output data of files.
    if not os.path.exists(sFilePath) : 
        os.mkdir(sFilePath)
    fo = open(sFilePath+filename, 'w+')
    fo.write(text)
    fo.close()

if __name__ == "__main__":
    path_tfidf = '../Training_Data/tfidffile/'
    VocabularyBase = path_tfidf + "VocabularyBase.txt"
    filename = "TotalBase.txt"
    if not os.path.exists(path_tfidf + filename):
        print("Can't find {0}", path)
    else:
        print(len(CreateVocabularyBase(path_tfidf, filename)))
    (allfile, path) = LookupIndex.lookupFileList("../Training_Data/segfile/")
    for ff in allfile :
        print("Filter by Vocabulary Base on " + ff)
        filterByVocab(path, ff, VocabularyBase)