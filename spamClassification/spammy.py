'''
Created on Aug 6, 2016

@author: eazymac25
'''
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import operator

class nBayes(object):
    def __init__(self):
        self.path = "C:\\Users\\eazymac25.KyleMacNeney-PC\\Downloads\\Spam Detection\\"
        self.spamFolder = 'spam'
        self.hamFolder = 'easy_ham'
        self.labelDict = defaultdict(int)
        self.wordCount = defaultdict(int)
        self.wordCountOnce = defaultdict(int)
        self.wordByLabel = defaultdict(lambda: defaultdict(int))
        self.wordByLabelOnce = defaultdict(lambda: defaultdict(int))
        self.tt = 0
        self.folders = [(self.spamFolder, 'SPAM'),
                   (self.hamFolder, 'HAM')]
    
    
    def buildData(self,pathName):
        for root, dir, filenames in os.walk(pathName):
            for f in filenames:
                fPath = os.path.join(root,f)
                if os.path.isfile(fPath):
                    fReader = open(fPath,'r',encoding='latin-1')
                    isbody= False
                    content = list()
                    for line in fReader:
                        if isbody:
                            content.append(line)
                        elif line =='\n': 
                            isbody = True
                    fReader.close()
                    messageBody = '\n'.join(content)
                    yield fPath, messageBody
                    
    
    def createDataFrame(self,pathName, classification):
        emails = []
        index = []
        for filePath, body in self.buildData(pathName):
            emails.append({'body':body,'type':classification})
            index.append(filePath)
        df = pd.DataFrame(emails,index=index)
        return df
    
    
    def counter(self,dframe):
        for index, row in dframe.iterrows():
            self.labelDict[row[1]]+= 1
            self.tt +=1
            tempWords = []
            for word in row[0].lower().split():
                if len(word)<20:
                    self.wordByLabel[word][row[1]]+=1
                    self.wordCount[word]+= 1
                    tempWords.append(word)
            tempWords = set(tempWords)
            for t in tempWords:
                self.wordByLabelOnce[t][row[1]]+=1
                self.wordCountOnce[word]+= 1
                
    
    def pOfWord(self,word,label):
        fCount = self.wordByLabelOnce[word][label]
        lCount = self.labelDict[label]
        
        if fCount and lCount:
            #print(float(fCount/lCount))
            return float(fCount/lCount)
        return 0
    
    def weightProb(self,feature,label, weight=1.0, ap = 0.5):
        iProb = self.pOfWord(feature,label)
        featureTotal = self.wordCountOnce[feature]
        if featureTotal>0:
            #print(float((weight*ap)+ (featureTotal*iProb))/(weight*featureTotal))
            return float((weight*ap)+ (featureTotal*iProb))/(weight*featureTotal)
        return 1
    
    def docProb(self,features, label):
        p =1 
        for feature in features:
            p *= self.weightProb(feature, label)
        return p
    
    def prob(self,features,label):
        if not self.tt:
            return 0
        
        lProb = float(self.labelDict[label])/self.tt
        dProb = self.docProb(features,label)
        return dProb*lProb
    
    def dclassy(self,features, limit=5):
        probs = {}
        for label in self.labelDict.keys():
            probs[label]= self.prob(features,label)
        #print (sorted(probs.items(), key=lambda v: v, reverse=True)[:limit])
        return sorted(probs.items(), key=lambda v:v[1], reverse=True)

if __name__ == '__main__':
    data = pd.DataFrame({'body':[],'type':[]})
    nbd = nBayes()
    for folder, classify in nbd.folders:
        data = data.append(nbd.createDataFrame(nbd.path+folder,classify))
        
    data = data.reindex(np.random.permutation(data.index))
    #print(data.head())
    nbd.counter(data)
    print(str(nbd.wordCountOnce['date']))
    #print(wordByLabel['president'])
    correct=0 
    mtotal = 0
    for index,row in data.iterrows():
        c = []
        for w in row[0].lower().split():
            if len(w) <20:
                c.append(w)
        temp = set(c)
        res =nbd.dclassy(temp)
        best = res[0][0]
        if best == row[1]:
            correct += 1
        mtotal += 1
    pct = float((correct)/mtotal)
    print(pct)
        
            
#     from sklearn.feature_extraction.text import CountVectorizer
#  
#     count_vectorizer = CountVectorizer()
#     counts = count_vectorizer.fit_transform(data['body'].values)
#     print(counts)