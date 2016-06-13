import os
import sys
import string
import time

libpath = '/Users/vick/Documents/Python/'
sys.path.append(libpath)
import TextPreprocess_Collection as TPPC
import LookupIndex

argv = "/Users/vick/Documents/Testing_Data/testfolder/"

if __name__ == "__main__":
    (allfile, path) = LookupIndex.lookupFileList(argv)
    for ff in allfile :
        print("Using stem on " + ff)
        TPPC.preprocessFile(ff, path)

    TPPC.Tfidf(allfile)