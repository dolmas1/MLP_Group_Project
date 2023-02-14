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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
id2label = {0: "noHate", 1: "hate"}

# Evaluation Function
def evaluate(model, test_loader, destination_path, model_name, tokenizer, model_type):
    """Evaluation function for testing purposes.

    Args:
        model: the initialized model to test
        test_loader: iterator for the test set
        destination_path (str): path where to store the results
        model_name (str): string how the models result shall be saved
    """
    y_pred = []; y_true = []; y_probs = []
    batch_tokens = []
    lig_scores = []

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
                toks = [tok.replace("Ġ", "") for tok in tokenizer.convert_ids_to_tokens(text[example]) if tok != tokenizer.pad_token]
                batch_tokens.append(toks)
                tokens.append(toks)
            
            # layerwise integrated gradient
            ligs = get_integrated_gradients_score(text, labels, tokens, tokenizer, model, model_type)
            for score in ligs:
                lig_scores.append(score.tolist())

            y_probs.extend(probs.tolist())
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())


    y_probs = np.array([prob[1] for prob in y_probs])
    y_true = np.array(y_true)

    result_table = PrettyTable(["Tokens", "Lime", "Shap", "Integrated Gradients", "Probability for Hate", "Prediction", "Predicted Label"])
    for tok, lig, prob, pred in zip(batch_tokens, lig_scores, y_probs, y_pred):
        result_table.add_row([tok, "", "", lig, round(prob, 2), id2label[pred], pred])
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