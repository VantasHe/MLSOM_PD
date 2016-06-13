import os
import sys
import string

def lookupFileList(path) :
    fileList = []
    files = os.listdir(path)
    for f in files :
        if (f[0] == '.') :
            pass
        else :
            fileList.append(f)
    return fileList, path

def lookupAllFile(rootPath):
    fileList = []
    for root, dirs, files in os.walk(rootPath):
        for name in dirs:
            if name[0] == '.':
                dirs.remove(name)
        for name in files:
            if name[0] == '.':
                pass
            else
                fileList.append(root+'/'+name)
    return fileList