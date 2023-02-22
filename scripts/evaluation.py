import logging
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
from prettytable import PrettyTable


from integrated_gradients import get_integrated_gradients_score
from attention import get_attention_scores

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
id2label = {0: "noHate", 1: "hate"}
#label2id = {"ungrammatical":0, "grammatical":1}

# Evaluation Function (for individual models)
def evaluate(model, test_loader, destination_path, model_name, tokenizer, model_type):
    """Evaluation function for testing purposes (single models only).

    Args:
        model: the initialized model to test
        test_loader: iterator for the test set
        destination_path (str): path where to store the results
        model_name (str): string how the models result shall be saved
    """
    y_pred = []; y_true = []; y_probs = []
    batch_tokens = []
    lig_scores = []; attention_scores = []

    model.eval()
    
    with torch.no_grad():
        
        for batch, batch_labels in test_loader:
        
            labels = batch_labels.to(device)
            text = batch['input_ids'].squeeze(1).to(device) 


            output = model(text, labels)

            logits = output.logits
            probs = F.softmax(logits, dim=1)

            # gets tokens for printing in file
            tokens = []
            for example in range(len(text)):
                toks = [tok for tok in tokenizer.convert_ids_to_tokens(text[example]) if tok != tokenizer.pad_token]
                batch_tokens.append(toks)
                tokens.append(toks)
            
            # attention scores
            attention_mask = batch['attention_mask'].to(device)
            attention_scores +=  get_attention_scores(attention_mask, output.attentions)

            # layerwise integrated gradient
            lig_scores += get_integrated_gradients_score(text, labels, tokens, tokenizer, model, model_type)



            y_probs.extend(probs.tolist())
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())


    y_probs = np.array([prob[1] for prob in y_probs])
    y_true = np.array(y_true)

    result_table = PrettyTable(["Tokens", "Lime", "Shap", "Attention", "Integrated Gradients", "Probability for Hate", "Predicted Label"])
    for tok, att, lig, prob, pred in zip(batch_tokens, attention_scores, lig_scores, y_probs, y_pred):
        result_table.add_row([tok, "", "",att,  lig, round(prob, 2),  pred])
    result_table.border = False
    result_table.align = "l"
    with open(os.path.join(destination_path, f"predictions_model_{model_name}.csv"), "w+") as csv_out:
        csv_out.write(result_table.get_csv_string())
    with open(os.path.join(destination_path, f"predictions_model_{model_name}.txt"), "w+") as txt_out:
        txt_out.write(str(result_table))

        


    logging.info('Classification Report:')
    logging.info('\n' + classification_report(y_true, y_pred, labels=[1,0], digits=4))
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=[1,0]).ravel()
    logging.info(f"True Positives: {tp}")
    logging.info(f"False Positives: {fp}")
    logging.info(f"True Negatives: {tn}")
    logging.info(f"False Negatives {fn}")

   
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['HATE', 'NO-HATE'])
    ax.yaxis.set_ticklabels(['HATE', 'NO-HATE'])
    plt.savefig(os.path.join(destination_path, f"{model_name}_heatmap.png"))
    plt.close()


# Evaluation function (for ensembles)
def evaluate_ensemble(constituent_models, constituent_model_names, test_loaders, destination_path, model_name, tokenizers, model_types, ensemble_method = 'majority'):
    """Evaluation function for testing purposes (ensembles).

    Args:
        constituent_models (list): the set of initialized models to include in the ensemble
        constituent_model_names (list): list of strings describing each constituent model (as model type may not be unique identifier)
        test_loaders (list): list of iterators for the test set (each iterator uses the correct tokenizer for the associated constituent model)
        destination_path (str): path where to store the results
        model_name (str): string of ensemble model name (how results shall be saved)
        tokenizers (list): the set of tokenizers used for each constituent model
        model_type (list): the set of model types (str) for each constituent model (taken from classifier['constituent_models']['embeddings'])
        ensemble_method (str): one of 'majority', 'wt_avg', 'latent', 'inter'
                               'majority': simple majority vote across predicted class labels of each constituent model 
                               'wt_avg': for each model_type category, create a new network whose weights are the avg of all constituent models in that category. Then run inference, and majority vote
                               'latent': for each model_type category, get avg latent embedding of test examples according to constituent models in that category. Build new classifier on resultant embeddings. Then run inference, and majority vote.
                               'inter': interpretability weighted ensemble (details TBC)
    """
    # For now, evaluate the first model only (will implement proper ensemble methods in next commit)
    model = constituent_models[0]
    tokenizer = tokenizers[0]
    model_type = model_types[0]
    test_loader = test_loaders[0]
    # end
    
    y_pred = []; y_true = []; y_probs = []
    batch_tokens = []
    lig_scores = []; attention_scores = []

    model.eval()
    
    
    with torch.no_grad():
        
        for batch, batch_labels in test_loader:
        
            labels = batch_labels.to(device)
            text = batch['input_ids'].squeeze(1).to(device) 


            output = model(text, labels)

            logits = output.logits
            probs = F.softmax(logits, dim=1)

            # gets tokens for printing in file
            tokens = []
            for example in range(len(text)):
                toks = [tok for tok in tokenizer.convert_ids_to_tokens(text[example]) if tok != tokenizer.pad_token]
                batch_tokens.append(toks)
                tokens.append(toks)
            
            # attention scores
            attention_mask = batch['attention_mask'].to(device)
            attention_scores +=  get_attention_scores(attention_mask, output.attentions)

            # layerwise integrated gradient
            lig_scores += get_integrated_gradients_score(text, labels, tokens, tokenizer, model, model_type)



            y_probs.extend(probs.tolist())
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())


    y_probs = np.array([prob[1] for prob in y_probs])
    y_true = np.array(y_true)

    result_table = PrettyTable(["Tokens", "Lime", "Shap", "Attention", "Integrated Gradients", "Probability for Hate", "Predicted Label"])
    for tok, att, lig, prob, pred in zip(batch_tokens, attention_scores, lig_scores, y_probs, y_pred):
        result_table.add_row([tok, "", "",att,  lig, round(prob, 2),  pred])
    result_table.border = False
    result_table.align = "l"
    with open(os.path.join(destination_path, f"predictions_model_{model_name}.csv"), "w+") as csv_out:
        csv_out.write(result_table.get_csv_string())
    with open(os.path.join(destination_path, f"predictions_model_{model_name}.txt"), "w+") as txt_out:
        txt_out.write(str(result_table))

        


    logging.info('Classification Report:')
    logging.info('\n' + classification_report(y_true, y_pred, labels=[1,0], digits=4))
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=[1,0]).ravel()
    logging.info(f"True Positives: {tp}")
    logging.info(f"False Positives: {fp}")
    logging.info(f"True Negatives: {tn}")
    logging.info(f"False Negatives {fn}")


    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['HATE', 'NO-HATE'])
    ax.yaxis.set_ticklabels(['HATE', 'NO-HATE'])
    plt.savefig(os.path.join(destination_path, f"{model_name}_heatmap.png"))
    plt.close()