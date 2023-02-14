# Libraries

import pandas as pd
import os
from torch.utils.data import Dataset
import numpy as np

label2id = {"noHate":0, "hate":1}

class HateSpeech(Dataset):

    pred_labels = ["noHate", "hate"]

    def __init__(self, root_dir, label_file, tokenizer):
        """Constructor method for sepsis data.
        Args:
            root_dir (str): path to the root directory
            label_file (str): filename of the csv file with examples labels
        """
        self.rootdir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, label_file), sep="\t", header=0)

        self.labels = [label2id[label] for label in self.df['label']]
        self.texts = [tokenizer(text, return_tensors = "pt", padding='max_length', 
                                truncation=True, max_length=512) for text in self.df['example']]
        
    def __len__(self):
        """Returns length based on the data frame length (= number of data points)
        Returns (int):  length based on the data frame length (= number of data points)
        """
        return len(self.df)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_texts, batch_labels

    def classes(self):
        return self.labels