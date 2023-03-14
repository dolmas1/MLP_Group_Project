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
from evaluation import evaluate



# Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("DEVICE USED: ", device)


transformers_logging.set_verbosity_error()

# deterministic behaviour
def set_seed(seed):
    """Sets all random seeds for deterministic behaviour.
    Args:
        seed (int): integer defining the random seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)




tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def run_cl(embeddings, model, interpretation, data, only_test=False):
    """Main function that runs the classifier model, trains and evaluates it.

    Args:
        embeddings (dict): dictionary containing descriptions of LMs for tokenizer and new LMs
        model (dict): dictionary that values contain information about model parameters
        data (dict): dictionary that values contains data paths
        lda (dict): dictionary that values contains (n)lda data paths
        explanations (int): number of attention explanations to output

    Raises:
        FileExistsError: if the destination path, where the model is stored, already exists
    """

    seed = model["random_seed"]
    set_seed(seed)

    # model parameter
    epochs = model["epochs"]
    positive_class_weight = model["positive_class_weight"]
    batch_size = model["batch_size"]
    destination_path = model["destination_path"]
    early_stopping = model["early_stopping"]

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
    embeds = embeddings["embeddings"]
    tok = embeddings["tokenizer"]
    if only_test:
        try:
            parameter_path = embeddings["load_from"]
        except:
            raise Exception("No path specified to load model from!")


    # model and tokenizer
    classifier = Classifier(embeds, positive_class_weight).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tok)

    # interpretation stuff
    analysis = interpretation["analysis"]

    # data parameter
    path = data["path"]
    train_file = data["train_file"]
    dev_file = data["dev_file"]
    test_file = data["test_file"]

    test_file_path = os.path.join(path, test_file)

    # load data

    if "hate" in path:
        train_hate_data = HateSpeech(root_dir=path, label_file=train_file, tokenizer=tokenizer)
        dev_hate_data = HateSpeech(root_dir=path, label_file=dev_file, tokenizer=tokenizer)
        test_hate_data = HateSpeech(root_dir=path, label_file=test_file, tokenizer=tokenizer)


        train_dataloader = torch.utils.data.DataLoader(train_hate_data, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(dev_hate_data, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_hate_data, batch_size=batch_size)
    
    elif "cola" in path:
        train_cola_data = CoLA(root_dir=path, label_file=train_file, tokenizer=tokenizer)
        dev_cola_data = CoLA(root_dir=path, label_file=dev_file, tokenizer=tokenizer)
        test_cola_data = CoLA(root_dir=path, label_file=test_file, tokenizer=tokenizer)


        train_dataloader = torch.utils.data.DataLoader(train_cola_data, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(dev_cola_data, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_cola_data, batch_size=batch_size)



    optimizer = optim.Adam(classifier.parameters(), lr=1e-6)

    if not only_test: 
        logging.info("Starting Training!")
        train(model=classifier, optimizer=optimizer, train_loader = train_dataloader, valid_loader = val_dataloader, num_epochs = epochs, 
                file_path = destination_path, early_stopping = early_stopping)
        logging.info("Finished Training!\n")

        # Evaluation
        plot_loss(destination_path, "metrics.pt")

        acc_checkpoint = os.path.join(destination_path, "model_best_acc.pt")
        loss_checkpoint = os.path.join(destination_path, "model_best_loss.pt")

        logging.info("\nEvaluation Model with best ACC")
        best_model_acc = Classifier(embeds, positive_class_weight).to(device)

        load_checkpoint(acc_checkpoint, best_model_acc)
        evaluate(best_model_acc, test_dataloader, destination_path, "best_acc", tokenizer, embeds,  test_file_path, analysis)
    
        logging.info("\nEvaluation Model with best LOSS")
        best_model_loss = Classifier(embeds, positive_class_weight).to(device)
        load_checkpoint(loss_checkpoint, best_model_loss)
        evaluate(best_model_loss, test_dataloader, destination_path, "best_loss", tokenizer, embeds, test_file_path, analysis)

    else:
        logging.info("Only testing, no training!")
        logging.info("Loading Model")
        loaded_model = Classifier(embeds, positive_class_weight).to(device)

        load_checkpoint(parameter_path, loaded_model)
        evaluate(loaded_model, test_dataloader, destination_path, "loaded_model", tokenizer, embeds, test_file_path, analysis)

    logging.info("\nGoodbye :)")
