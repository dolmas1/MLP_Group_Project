# Libraries


# import matplotlib.pyplot as plt
import torch



# Models

import torch.nn as nn
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    """
    Classifier Model for predictions, using pretrained BertForSequenceClassification or RobertaForSequenceClassification.
    """

    def __init__(self, embedding_model, positive_class_weight = 1.):
        """Init method for the classifier model.
        Args:
             embedding_model (str): path to the embeddings, either from HuggingFace or pretrained
            positive_class_weight (float): float defining the importance of the positive class for the loss value
        """
        super(Classifier, self).__init__()
     
        #self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1., float(positive_class_weight)]))
        self.loss = nn.CrossEntropyLoss()
        self.config = AutoConfig.from_pretrained(embedding_model, num_labels=2, output_hidden_states=True, output_attentions=True, return_dict=True)
        
        logging.info(f"Model used: {embedding_model}")
        if "roberta" in embedding_model:
            self.encoder = RobertaForSequenceClassification.from_pretrained(embedding_model, config=self.config)

        elif "bert" in embedding_model:        
            self.encoder = BertForSequenceClassification.from_pretrained(embedding_model, config=self.config)

        else:
            raise Exception(f"Architecture {embedding_model} not supported!")



    def forward(self, input_ids=None, token_type_ids=None, label=None, lig=False, shap=False, attention_mask=None):
        """Forward pass of the model of a given text.

        Args:
            text (input_ids): tensor containing the input_ids of the text to classify
            label (int): 1 (positive) or 0 (negative) labels for the given text, default: None

        Returns:
            out (Attributes): Attributes wrapper class with information important for further processing
        """
        text = input_ids

        #if token_type_ids !=None:
        #    text = token_type_ids

        if lig:
            text.to(device)
            return self.encoder(text, labels=label)
        
        if shap or attention_mask != None: return self.encoder(text, labels=label)

        text.to(device)

        if label != None:
            label.to(device)

        out = self.encoder(text, labels=label)

        l = self.loss(out[1], label)
       
        out = Attributes(logits=out[1], loss=l, hidden_states=out[2][-1][:,0], attentions=out["attentions"])


        return out

class Attributes(object):
    """Wrapper class containing information about the classifer output.
    """
    def __init__(self, logits = None, loss = None, hidden_states = None, attentions = None):
        """Init method of the Attributes class.

        Args:
            logits: the logits of the model
            loss: the loss of the output (only if label is provided)
            hidden_states: last hidden states of the model
            attentions: attention weights of the model
        """
        self.logits = logits
        self.loss = loss
        self.hidden_states = hidden_states
        self.attentions = attentions
