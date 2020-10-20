dev = open('DEV.txt', 'r')
dev_questions = open('DEV-questions.txt', 'w')
dev_labels = open('DEV-labels.txt', 'w')

questions = []
labels = []
lines = dev.readlines()

for line in lines:
    q = line.partition(" ")[2]
    questions.append(q)
    l = line.partition(" ")[0]
    l = l + "\n"
    labels.append(l)

dev_questions.writelines(questions)
dev_labels.writelines(labels)

dev.close()
dev_questions.close()
dev_labels.close()
