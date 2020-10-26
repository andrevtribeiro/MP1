import sys
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
import math
from multiprocessing import Pool

train={}
dev_questions = []

important_words= set()

weights_med=[-1,1,1]

def divideTrain(train_file,coarse):
    global train
    f=open(train_file,"r")
    lines = f.readlines()
    for line in lines:
        q = line.partition(" ")[2]
        l = line.partition(" ")[0]
        if coarse==True:
            l=l.partition(':')[0]
        l = l + "\n"
        if l not in train:
            train[l]=[]
        train[l].append(q)
    f.close()

def readDevQuestions(dev):
    global dev_questions
    f=open(dev,"r")
    dev_questions=f.readlines()
    f.close()

def preProcessingQuestions():
    global train, dev_questions
    stop_words = set(stopwords.words('english'))
    stop_words.add('?')
    stop_words.add(',') 
    ps = PorterStemmer()
    for label in train.keys():
        filtered_questions=[]
        for question in train[label]:
            word_tokens = word_tokenize(question)
            filtered_questions.append([ps.stem(w) for w in word_tokens if not w in stop_words])
        train[label] = filtered_questions
    filtered_questions=[]
    for question in dev_questions:
        word_tokens = word_tokenize(question)
        filtered_questions.append([ps.stem(w) for w in word_tokens if not w in stop_words])
      
    dev_questions = filtered_questions

    
def tfIdf():
    global train, important_words
    
    idfdic={}
    tfdic={}
    for label in train:
        tfdic[label]={}
        contador=0
        for question in train[label]:
            for word in question:
                if word not in tfdic[label]:
                    tfdic[label][word]=0
                tfdic[label][word]+=1
                contador+=1
        for word in tfdic[label]:
            if word not in idfdic:
                idfdic[word]=0
            idfdic[word]+=1
            tfdic[label][word]/=contador
        

    for label in tfdic:
        for word in tfdic[label]:
            tfdic[label][word]*=math.log(len(tfdic)/idfdic[word])
        tfdic[label] = sorted(tfdic[label].items(), key=lambda x: x[1], reverse=True)[:10]
        for tuplo in tfdic[label]:
            important_words.add(tuplo[0])
              
  
def med(dev_question,train_question):
    global important_words, weights_med
    lenDev=len(dev_question)
    lenTrain=len(train_question)

    matrix=[[0 for i in range(lenDev+1)] for j in range(lenTrain+1)]
    matrix[0]=[-i for i in range(lenDev+1)]
    for i in range(lenTrain+1):
        matrix[i][0]=-i

    j=1
    while j != lenDev+1:
        i=1
        while i!=lenTrain+1:
            if train_question[i-1]==dev_question[j-1]:
                matrix[i][j]=matrix[i-1][j-1]+ weights_med[1]
                if train_question[i-1] in important_words:
                    matrix[i][j]+= weights_med[2]
            else:
                matrix[i][j]=max(matrix[i-1][j],matrix[i][j-1],matrix[i-1][j-1])+ weights_med[0]
            i+=1
        j+=1
    return matrix[lenTrain][lenDev]

    
def med_question(dev_question):
    global train
    maximo=[-1,{}]
    for label in train.keys():
        for train_question in train[label]:
            new_max=med(dev_question,train_question)
            if maximo[1]=={} or maximo[0]<new_max:
                maximo[0]=new_max
                maximo[1]={label:1}
            elif maximo[0]==new_max:
                if label not in maximo[1]:
                    maximo[1][label]=0
                maximo[1][label]+=1  
    maxi=[-1,""]
    for label in maximo[1].keys(): 
        if maxi[1]=="" or maximo[1][label]>maxi[0]:
            maxi[0]=maximo[1][label]
            maxi[1]=label
    return maxi[1]
    
def coarse_model():
    global dev_questions, train, weights_med
    weights_med=[-1,2,5]
    output=""
    pool=Pool()
    results=pool.map(med_question,dev_questions)
    for result in results:
        output+=result  
    print(output)

def fine_model():
    global dev_questions, train, weights_med
    weights_med=[-2,3,5]
    output=""
    pool=Pool()
    results=pool.map(med_question,dev_questions)
    for result in results:
        output+=result  
    print(output)

def main(argv):
    train=argv[1]
    dev=argv[2]
    
    divideTrain(train,argv[0]=="-coarse")
    readDevQuestions(dev)

    preProcessingQuestions()
    tfIdf()
    if argv[0]=="-coarse":
        coarse_model()
    elif argv[0]=="-fine":
        fine_model()
    else:
        raise Exception("Option must be coarse or fine")



if __name__ == "__main__":
   main(sys.argv[1:])