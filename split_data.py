import os
import csv


# train = 1914
# test = 239
# dev = 239
HEADER = ['example', 'label']

def get_labels(label_file):
    file_label_dict = dict()

    with open (label_file, "r") as label_file:
        for index, line in enumerate(label_file):
            if index != 0:
                line = line.split(",")
                file_id = line[0]
                label = line[-1].replace("\n", "")
                file_label_dict[file_id] = label
    return file_label_dict


def get_row(label, file_name, source_data_dir):
    with open(source_data_dir + "/" + file_name) as speech_doc:
        sentence = speech_doc.readline()
    return [sentence, label]

def do_train_data(source_data_dir_train, train_dir, file_label_dict):
    
    with open(train_dir, 'w', encoding='UTF8', newline='') as train_csv:
        writer = csv.writer(train_csv,   delimiter='\t')
        writer.writerow(HEADER)

        for file in os.listdir(source_data_dir_train):
            file_name = os.fsdecode(file)
            file_id = file_name.replace(".txt", "")
            label = file_label_dict[file_id]              
            writer.writerow(get_row(label, file_name, source_data_dir_train))

def do_dev_test_data(source_data_dir_test, dev_dir, test_dir, file_label_dict):

    hate_counter = 0
    no_hate_counter = 1

    with open(dev_dir, 'w', encoding='UTF8', newline='') as dev_csv:
        dev_writer = csv.writer(dev_csv,  delimiter='\t')
        dev_writer.writerow(HEADER)

        with open(test_dir, 'w', encoding='UTF8', newline='') as test_csv:
            test_writer = csv.writer(test_csv,  delimiter='\t')
            test_writer.writerow(HEADER)

            for file in os.listdir(source_data_dir_test):
                file_name = os.fsdecode(file)
                file_id = file_name.replace(".txt", "")
                label = file_label_dict[file_id]  

                if label == "noHate": 
                    no_hate_counter += 1

                    if no_hate_counter <= 120: dev_writer.writerow(get_row(label, file_name, source_data_dir_test))                      
                    else: test_writer.writerow(get_row(label, file_name, source_data_dir_test))  

                else: 
                    hate_counter += 1

                    if hate_counter <= 120: dev_writer.writerow(get_row(label, file_name, source_data_dir_test)) 
                    else: test_writer.writerow(get_row(label, file_name, source_data_dir_test)) 

if __name__ == "__main__": 
    label_dir = "data/hatespeech/a_original/annotations_metadata.csv"
    source_data_dir_train = "data/hatespeech/a_original/sampled_train_original"
    source_data_dir_test = "data/hatespeech/a_original/sampled_test_original"
    train_dir = "data/hatespeech/train.csv"
    dev_dir = "data/hatespeech/dev.csv"
    test_dir = "data/hatespeech/test.csv"

    file_label_dict = get_labels(label_dir)
    do_train_data(source_data_dir_train, train_dir, file_label_dict)
    do_dev_test_data(source_data_dir_test, dev_dir, test_dir, file_label_dict)
