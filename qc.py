import sys
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



train_questions = []
train_labels = []
dev_questions = []
important_words_coarse = ["how","what","who","when","whom","where","which","stand","why"]
important_expressions=["how many","how does","how old","how long","how is","how can","how do","how much","for a living"]
important_words_fine = ["city","country","mountain","state","stand","why","year","date","month","day","name",
                        "group","company","team","title","job","occupation","profession","animal","creature"]
def divideTrain(train):
    global train_questions
    f=open(train,"r")
    lines = f.readlines()
    for line in lines:
        q = line.partition(" ")[2]
        train_questions.append(q)
        l = line.partition(" ")[0]
        l = l + "\n"
        train_labels.append(l)
    f.close()

def readDevQuestions(dev):
    global dev_questions
    f=open(dev,"r")
    dev_questions=f.readlines()
    f.close()

def preProcessingQuestions():
    global train_questions, dev_questions
    stop_words = set(stopwords.words('english'))
    stop_words.add('?')
    filtered_questions=[]
    for question in train_questions:
        word_tokens = word_tokenize(question)
        filtered_questions.append([w for w in word_tokens if not w in stop_words])
    train_questions = filtered_questions
    filtered_questions=[]
    for question in dev_questions:
        word_tokens = word_tokenize(question)
        filtered_questions.append([w for w in word_tokens if not w in stop_words])
        
    dev_questions = filtered_questions
        
        



def jaccard(dev_question,train_question):
    dev_question=set(dev_question)
    train_question=set(train_question)
    return len(dev_question.intersection(train_question))/len(dev_question.union(train_question))
    

def jaroDistance(dev_question,train_question):
    lenDev=len(dev_question)
    lenTrain=len(train_question)

    maxDist=min(lenDev,lenTrain)//2

    match=0

    hash_dev = [0] * lenDev
    hash_train = [0] * lenTrain

    for i in range(lenDev):
        for j in range(max(0,i-maxDist),min(lenTrain,i+maxDist+1)):
            
            if(dev_question[i]==train_question[j] and hash_train[j]==0):
                hash_dev[i]=1
                hash_train[j]=1
                match+=1
                break
    if match==0:
        return 0

    t=0
    index_train=0

    for i in range(lenDev):
        if hash_dev[i]==1:
            while hash_train[index_train]==0:
                index_train+=1
            if hash_dev[i] != hash_train[index_train]:
                index_train+=1
                t+=1

    return ( match/lenDev + match/lenTrain + (match-t/2)/match )/3 

    
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
                matrix[i][j]=matrix[i-1][j-1]
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
    global dev_questions, train_labels, train_questions
    output=""
    for dev_question in dev_questions:
        maximo=[-1,-1]
        maximo[0]=med(dev_question,train_questions[0])
        maximo[1]=0
        for tq_index in range(1,len(train_questions)):
            #new_max=jaccard(dev_question,train_questions[tq_index])
            #new_max=jaroDistance(dev_question,train_questions[tq_index])
            new_max=med(dev_question,train_questions[tq_index])
            if new_max>maximo[0]:
                maximo[0]=new_max
                maximo[1]=tq_index
        output+=train_labels[maximo[1]].partition(':')[0]+'\n'
    print(output)

def fine_model():
    global dev_questions, train_labels, train_questions
    output=""
    for dev_question in dev_questions:
        maximo=[-1,-1]
        maximo[0]=med(dev_question,train_questions[0])
        maximo[1]=0
        for tq_index in range(1,len(train_questions)):
            #new_max=jaccard(dev_question,train_questions[tq_index])
            #new_max=jaroDistance(dev_question,train_questions[tq_index])
            new_max=med(dev_question,train_questions[tq_index])
            if new_max>maximo[0]:
                maximo[0]=new_max
                maximo[1]=tq_index
        output+=train_labels[maximo[1]]
    print(output)

def main(argv):
    train=argv[1]
    dev=argv[2]
    
    divideTrain(train)
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