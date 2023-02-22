# Libraries

import logging
import torch
import os

from transformers import AutoTokenizer
from transformers import logging as transformers_logging
import torch.optim as optim



# Evaluation

import numpy as np
import random

# import other scripts

from classifiers import Classifier
from datasets import HateSpeech, CoLA
from training import train
from plot_loss import plot_loss
from checkpoints import load_checkpoint
from evaluation import evaluate_ensemble
from model import set_seed


# Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("DEVICE USED: ", device)


transformers_logging.set_verbosity_error()



def run_ensemble(constituent_models, data, ensemble_details):
    """Main function that collects pre-trained models, constructs an ensemble, and evaluates it.

    Args:
        constituent_models (list of dicts): list of constituent models, each specified by a dictionary containing descriptions of LMs for tokenizer and new LMs
        data (dict): dictionary that values contains data paths
        ensemble_details (dict): dictionary containing information of the ensemble to be built, including:
                                    random_seed (int): random seed
                                    destination_path (str): where will ensemble results be saved
                                    ensemble_method (str): one of 'majority', 'wt_avg', 'latent', 'inter'
                                           'majority': simple majority vote across predicted class labels of each constituent model 
                                           'wt_avg': for each model_type category, create a new network whose weights are the avg of all constituent models in that category.
                                                     Then run inference, and majority vote
                                           'latent': for each model_type category, get avg latent embedding of test examples according to constituent models in that category.
                                                     Build new classifier on resultant embeddings. Then run inference, and majority vote.
                                           'inter': interpretability weighted ensemble (details TBC)

    Raises:
        FileExistsError: if the destination path, where the model is stored, already exists
    """

    destination_path = ensemble_details['destination_path']
    seed = ensemble_details['random_seed']
    batch_size = data['batch_size']
    set_seed(seed)

    # check destination path and create directory
    if os.path.exists(destination_path):
        if len(os.listdir(destination_path)) > 1: raise FileExistsError(f"Model directory {destination_path} exists.")

    os.mkdir(destination_path)
    logging.basicConfig(level=logging.INFO, 
            handlers=[
                logging.FileHandler(os.path.join(destination_path, "run.log")),
                logging.StreamHandler()
                ]
            )
    logging.info("Welcome :)\n")
    logging.info(f"Device used: {device}\n")

        
    # pretrained models
    embeds = [model["embeddings"] for model in constituent_models]
    pos_class_wts = [model["positive_class_weight"] for model in constituent_models]
    toks = [model["tokenizer"] for model in constituent_models]
    try:
        parameter_paths = [model["load_from"] for model in constituent_models]
    except:
        raise Exception("No path specified to load model from!")


    # model and tokenizer
    tokenizers = [AutoTokenizer.from_pretrained(tok) for tok in toks]
    
    
    logging.info("Only testing, no training!")
    logging.info(f"Loading {len(constituent_models)} Constituent models")
    loaded_models = [Classifier(embed, pos_class_wts[i]).to(device) for i, embed in enumerate(embeds)]

    for i, model in enumerate(loaded_models):
        load_checkpoint(parameter_paths[i], model)
        
        
    # data parameter
    path = data["path"]
    train_file = data["train_file"]
    dev_file = data["dev_file"]
    test_file = data["test_file"]
    
    # load data
    if "hate" in path:
        train_data_arr = [HateSpeech(root_dir=path, label_file=train_file, tokenizer=tok) for tok in tokenizers]
        dev_data_arr = [HateSpeech(root_dir=path, label_file=dev_file, tokenizer=tok) for tok in tokenizers]
        test_data_arr = [HateSpeech(root_dir=path, label_file=test_file, tokenizer=tok) for tok in tokenizers]
        
    elif "cola" in path:
        train_data_arr = [CoLA(root_dir=path, label_file=train_file, tokenizer=tok) for tok in tokenizers]
        dev_data_arr = [CoLA(root_dir=path, label_file=dev_file, tokenizer=tok) for tok in tokenizers]
        test_data_arr = [CoLA(root_dir=path, label_file=test_file, tokenizer=tok) for tok in tokenizers]

    train_dataloaders = [torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) for train_data in train_data_arr]
    val_dataloaders = [torch.utils.data.DataLoader(dev_data, batch_size=batch_size) for dev_data in dev_data_arr]
    test_dataloaders = [torch.utils.data.DataLoader(test_data, batch_size=batch_size) for test_data in test_data_arr]
        
    evaluate_ensemble(constituent_models = loaded_models,
                      test_loaders = test_dataloaders,
                      destination_path = destination_path,
                      model_name = "ensemble_model",
                      tokenizers = tokenizers,
                      model_types = embeds,
                      ensemble_method = ensemble_details['ensemble_method'])

    logging.info("\nGoodbye :)")