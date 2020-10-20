import sys
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



train={}
dev_questions = []
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
    
    for label in train.keys():
        filtered_questions=[]
        for question in train[label]:
            word_tokens = word_tokenize(question)
            filtered_questions.append([w for w in word_tokens if not w in stop_words])
        train[label] = filtered_questions
    filtered_questions=[]
    for question in dev_questions:
        word_tokens = word_tokenize(question)
        filtered_questions.append([w for w in word_tokens if not w in stop_words])
        
    dev_questions = filtered_questions
    
        


    
def med(dev_question,train_question):
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
                matrix[i][j]=matrix[i-1][j-1]+2
                '''if train_question[i-1].lower() in important_words_coarse:
                    matrix[i][j]+=1
                if train_question[i-1].lower() in important_words_fine:
                    matrix[i][j]+=2'''

            else:
                matrix[i][j]=max(matrix[i-1][j],matrix[i][j-1],matrix[i-1][j-1])-1
            i+=1
        j+=1
    '''for expression in important_expressions:
        if expression in dev_question:
            matrix[lenTrain][lenDev]+=10'''
    return matrix[lenTrain][lenDev]

    


def coarse_model():
    global dev_questions, train
    output=""
    for dev_question in dev_questions:
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
        output+=maxi[1]
    print(output)

def fine_model():
    global dev_questions, train
    output=""
    for dev_question in dev_questions:
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
        output+=maxi[1]
    print(output)

def main(argv):
    train=argv[1]
    dev=argv[2]
    
    divideTrain(train,argv[0]=="-coarse")
    readDevQuestions(dev)

    preProcessingQuestions()
    if argv[0]=="-coarse":
        coarse_model()
    elif argv[0]=="-fine":
        fine_model()
    else:
        raise Exception("Option must be coarse or fine")



if __name__ == "__main__":
   main(sys.argv[1:])