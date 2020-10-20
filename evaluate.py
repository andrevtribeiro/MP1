predicted_labels_file = open('predicted-labels.txt', 'r')
dev_labels_file = open('DEV-labels.txt', 'r')

predicted_labels = predicted_labels_file.readlines()
dev_labels=dev_labels_file.readlines()

coarse_flag=False
if ':' not in predicted_labels[0]:
    coarse_flag=True

counter=0
for predicted_label, dev_label in zip(predicted_labels,dev_labels):
    if coarse_flag and predicted_label[:-1]==dev_label.partition(':')[0]:
        counter+=1
    elif not coarse_flag and predicted_label==dev_label:
        counter+=1

print(counter/len(predicted_labels))

predicted_labels_file.close()
dev_labels_file.close()