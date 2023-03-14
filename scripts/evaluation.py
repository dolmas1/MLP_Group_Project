import logging
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import re
from prettytable import PrettyTable


from integrated_gradients import get_integrated_gradients_score
from attention import get_attention_scores
from lime_shap import get_shap_scores, get_lime_scores
from detokenize import detokenize_single, force_agreement

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
id2label = {0: "noHate", 1: "hate"}
#label2id = {"ungrammatical":0, "grammatical":1}

# Evaluation Function (for individual models)
def evaluate(model, test_loader, destination_path, model_name, tokenizer, model_type, test_file_path, analysis):
    """Evaluation function for testing purposes (single models only).

    Args:
        model: the initialized model to test
        test_loader: iterator for the test set
        destination_path (str): path where to store the results
        model_name (str): string how the models result shall be saved
    """
    if analysis:
        if "hate" in path:
            df = pd.read_csv(test_file_path, sep="\t", header=0)
        elif "cola" in path:
            columns = ["id", "label", "star", "example"]
            df = pd.read_csv(test_file_path, sep="\t", header=None, names=columns)

        original_text = [text for text in df['example']]
        original_text_id = 0

    y_pred = []; y_true = []; y_probs = []
    batch_tokens = []
    lig_scores = []; attention_scores = []; shap_scores = []; lime_scores = []

    model.eval()
    
    with torch.no_grad():
        
        for batch, batch_labels in test_loader:
        
            labels = batch_labels.to(device)
            text = batch['input_ids'].squeeze(1).to(device) 
            
            output = model(text, label=labels)

            logits = output.logits
            probs = F.softmax(logits, dim=1)

            if analysis:
                # get original text for interpretability methods
                original_batch_text = original_text[original_text_id : original_text_id + len(labels)]
                original_text_id += len(labels) 


                # gets tokens for printing in file
                tokens = []
                for example in range(len(text)):
                    toks = [tok for tok in tokenizer.convert_ids_to_tokens(text[example]) if tok != tokenizer.pad_token]
                    batch_tokens.append(toks)
                    tokens.append(toks)


            # lime scores
            lime_scores += get_lime_scores(model, text, tokenizer)

            # shap scores
            shap_scores += get_shap_scores(model, text, tokenizer, original_batch_text)
            
            # attention scores
            attention_mask = batch['attention_mask'].to(device)
            attentions = output.attentions
            attention_scores +=  get_attention_scores(attention_mask, attentions)

                # layerwise integrated gradient
                lig_scores += get_integrated_gradients_score(text, labels, tokens, tokenizer, model, model_type)

                y_probs.extend(probs.tolist())
                y_pred.extend(torch.argmax(logits, 1).tolist())
                y_true.extend(labels.tolist())

                assert len(lig_scores) == len(attention_scores) == len(lime_scores) == len(shap_scores)
    y_probs = np.array([prob[1] for prob in y_probs])
    y_true = np.array(y_true)

    if analysis:
        result_table = PrettyTable(["Tokens", "Lime", "Shap", "Attention", "Integrated Gradients", "Probability for Hate", "Predicted Label"])
        for tok, lime, shap, att, lig, prob, pred in zip(batch_tokens, lime_scores, shap_scores, attention_scores, lig_scores, y_probs, y_pred):
            result_table.add_row([tok, lime, shap, att,  lig, round(prob, 2),  pred])
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
def evaluate_ensemble(constituent_models, constituent_model_names, test_loaders, destination_path, model_name, tokenizers, model_types, test_file_path, ensemble_method = 'majority'):
    """Evaluation function for testing purposes (ensembles).

    Args:
        constituent_models (list): the set of initialized models to include in the ensemble
        constituent_model_names (list): list of strings describing each constituent model (as model type may not be unique identifier)
        test_loaders (list): list of iterators for the test set (each iterator uses the correct tokenizer for the associated constituent model)
        destination_path (str): path where to store the results
        model_name (str): string of ensemble model name (how results shall be saved)
        tokenizers (list): the set of tokenizers used for each constituent model
        model_type (list): the set of model types (str) for each constituent model (taken from classifier['constituent_models']['embeddings'])
        test_file_path (str): path to test file (needed to load original sentence strings for LIME/SHAP)
        ensemble_method (str): one of 'majority', 'wt_avg', 'latent', 'inter'
                               'majority': simple majority vote across predicted class labels of each constituent model 
                               'wt_avg': for each model_type category, create a new network whose weights are the avg of all constituent models in that category. Then run inference, and majority vote
                               'latent': for each model_type category, get avg latent embedding of test examples according to constituent models in that category. Build new classifier on resultant embeddings. Then run inference, and majority vote.
                               'inter': interpretability weighted ensemble (details TBC)
    """
    
    if ensemble_method in ['majority', 'inter']:

        model_id = []; test_obs_idx = []
        y_pred = []; y_true = []; y_probs = []
        batch_tokens = []
        lig_scores = []; attention_scores = []; shap_scores = []; lime_scores = []

        num_models = len(constituent_models)

        # loop through constituent models
        for i in range(num_models):
            model = constituent_models[i]
            constituent_model_name = constituent_model_names[i]
            tokenizer = tokenizers[i]
            model_type = model_types[i]
            test_loader = test_loaders[i]

            df = pd.read_csv(test_file_path, sep="\t", header=0)
            original_text = [text for text in df['example']]
            original_text_id = 0
            
            logging.info(f"\nStarting analysis for {constituent_model_name}")
            
            with torch.no_grad():

                for batch, batch_labels in test_loader:

                    labels = batch_labels.to(device)
                    text = batch['input_ids'].squeeze(1).to(device)
                    
                    # get original text for interpretability methods
                    original_batch_text = original_text[original_text_id : original_text_id + len(labels)]
                    original_text_id += len(labels)

                    output = model(text, label=labels)

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
                    
                    # lime scores
                    lime_scores += get_lime_scores(model, text, tokenizer)

                    # shap scores
                    shap_scores += get_shap_scores(model, text, tokenizer, original_batch_text)
                    

                    y_probs.extend(probs.tolist())
                    y_pred.extend(torch.argmax(logits, 1).tolist())
                    y_true.extend(labels.tolist())

                    model_id.extend(len(labels) * [constituent_model_name])
                    
                    assert len(lig_scores) == len(attention_scores) == len(lime_scores) == len(shap_scores)

        # collect constituent model predictions, produce csv
        logging.info(f"\nSaving all constituent model outputs")
        
        y_probs = np.array([prob[1] for prob in y_probs])
        y_true = np.array(y_true)

        test_obs_idx.extend(num_models * list(range(int(len(y_true) / num_models))))

        result_table = pd.DataFrame({"Model_id": model_id,
                                     "Test_example": test_obs_idx,
                                     "Tokens": batch_tokens,
                                     "Lime": lime_scores,
                                     "Shap": shap_scores,
                                     "Attention": attention_scores,
                                     "Integrated Gradients": lig_scores,
                                     "Probability for Hate": y_probs,
                                     "Predicted Label": y_pred,
                                     "True Label": y_true})
        

        # combine the predictions
        if ensemble_method == 'majority':

            result_table.to_csv(os.path.join(destination_path, f"all_constituent_predictions_{model_name}.csv"), index = False)

            # save majority vote predictions
            ensemble_preds = pd.DataFrame({'Test_example': result_table['Test_example'][:int(len(y_true) / num_models)],
                                       'Ensemble_pred': result_table.groupby('Test_example')['Predicted Label'].agg(lambda x: pd.Series.mode(x)[0]),
                                       'True_label': result_table.groupby('Test_example')['True Label'].agg(lambda x: pd.Series.mode(x)[0])})
            ensemble_preds.to_csv(os.path.join(destination_path, f"ensemble_predictions_{model_name}.csv"), index = False)


        elif ensemble_method == 'inter':
            
            # FIRST step of unifying tokens (and associated interpretability scores): process each row individually
            merged_tokens = [detokenize_single(batch_tokens[i],
                                               np.array([lime_scores[i], shap_scores[i], attention_scores[i], lig_scores[i]]),
                                               model_id[i]) for i in range(len(test_obs_idx))]
            
            # SECOND step of unifying tokens and scores: (forces agreement across full set of outputs for each test observation)
            unified_results = [i for i in range(len(test_obs_idx))]
            
            # loop through each test observation
            for test_obs in set(test_obs_idx):

                # collect all tokens/inter_scores associated with this observation (= one item for each constituent model)
                idx = [i for i, x in enumerate(test_obs_idx) if x==test_obs]
                merged_toks_list = [merged_tokens[i] for i in idx]
                num_models = len(merged_toks_list)

                # forward pass to force agreement
                fwd_pass = [merged_toks_list[0]]
                for i in range(1, num_models):
                    op = force_agreement(toks_a_input = fwd_pass[0][0],
                                         toks_b_input = merged_toks_list[i][0],
                                         inter_a = fwd_pass[0][1],
                                         inter_b = merged_toks_list[i][1])
                    fwd_pass[0] = [op[0], op[1]]
                    fwd_pass.append([op[0], op[2]])

                # backwards pass to force agreement
                bkw_pass = [fwd_pass[-1]]
                for i in range(num_models-1, 0, -1):
                    op = force_agreement(toks_a_input = bkw_pass[0][0],
                                         toks_b_input = fwd_pass[i-1][0],
                                         inter_a = bkw_pass[0][1],
                                         inter_b = fwd_pass[i-1][1])
                    bkw_pass[0] = [op[0], op[1]]
                    bkw_pass.append([op[0], op[2]])

                bkw_pass.reverse()

                # check all models now have the same number of tokens
                assert min([len(a[0]) for a in bkw_pass]) == max([len(a[0]) for a in bkw_pass]), 'unification has failed, please debug'

                # calculate model agreement scores
                lime_scores = [a[1][0, :] for a in bkw_pass]
                shap_scores = [a[1][1, :] + 1e-8 for a in bkw_pass] # sometimes SHAP are all zero, which causes divide-by-zero errors
                attn_scores = [a[1][2, :] for a in bkw_pass]
                intg_scores = [a[1][3, :] for a in bkw_pass]
                lime_agreement = [sum([cos_sim(a, b) for b in lime_scores])/len(lime_scores) for a in lime_scores]
                shap_agreement = [sum([cos_sim(a, b) for b in shap_scores])/len(shap_scores) for a in shap_scores]
                attn_agreement = [sum([cos_sim(a, b) for b in attn_scores])/len(attn_scores) for a in attn_scores]
                intg_agreement = [sum([cos_sim(a, b) for b in intg_scores])/len(intg_scores) for a in intg_scores]
                
                # keep track of results in the right order
                for i, j in enumerate(idx):
                    unified_results[j] = {'Tokens_unified': bkw_pass[i][0],
                                          'Inter_scores_unified': bkw_pass[i][1],
                                          'Lime_agreement': lime_agreement[i],
                                          'Shap_agreement': shap_agreement[i],
                                          'Attention_agreement': attn_agreement[i],
                                          'Integrated_Grad_agreement':  intg_agreement[i]}
            
            # Save the unified tokens / interpretability scores / agreement scores
            result_table['Tokens_unified'] = [a['Tokens_unified'] for a in unified_results]
            result_table['Lime_unified'] = [a['Inter_scores_unified'][0, :] for a in unified_results]
            result_table['Shap_unified'] = [a['Inter_scores_unified'][1, :] for a in unified_results]
            result_table['Attention_unified'] = [a['Inter_scores_unified'][2, :] for a in unified_results]
            result_table['Integrated_Grad_unified'] = [a['Inter_scores_unified'][3, :] for a in unified_results]
            result_table['Lime_agreement'] = [a['Lime_agreement'] for a in unified_results]
            result_table['Shap_agreement'] = [a['Shap_agreement'] for a in unified_results]
            result_table['Attention_agreement'] = [a['Attention_agreement'] for a in unified_results]
            result_table['Integrated_Grad_agreement'] = [a['Integrated_Grad_agreement'] for a in unified_results]
            
            result_table['Overall_agreement'] = result_table['Lime_agreement'] + result_table['Shap_agreement'] + result_table['Attention_agreement'] + result_table['Integrated_Grad_agreement']
            
            result_table.to_csv(os.path.join(destination_path, f"all_constituent_predictions_{model_name}.csv"), index = False)
            result_table.to_pickle(os.path.join(destination_path, f"all_constituent_predictions_{model_name}.pkl"))
            
            
            # make the final ensemble predictions
            ensemble_preds = result_table.loc[result_table.groupby("Test_example")["Overall_agreement"].idxmax()]
            ensemble_preds = ensemble_preds[['Model_id', 'Test_example', 'Predicted Label', 'True Label']]
            ensemble_preds = ensemble_preds.rename(columns={'Predicted Label': 'Ensemble_pred', 'True Label': 'True_label'})

            # save the final predictions
            ensemble_preds.to_csv(os.path.join(destination_path, f"ensemble_predictions_{model_name}.csv"), index = False)
            


        # Calculate and save ensemble metrics
        y_true = np.array(ensemble_preds['True_label'])
        y_pred = np.array(ensemble_preds['Ensemble_pred'])

        logging.info('Classification Report:')
        logging.info('\n' + classification_report(y_true, y_pred, labels=[1,0], digits=4))
        tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=[1,0]).ravel()
        logging.info(f"True Positives: {tp}")
        logging.info(f"False Positives: {fp}")
        logging.info(f"True Negatives: {tn}")
        logging.info(f"False Negatives {fn}")

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * tp / (2*tp + fp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f'precision = {round(100 * precision, 2)}')
        print(f'recall = {round(100 * recall, 2)}')
        print(f'f1_score = {round(100 * f1_score, 2)}')
        print(f'accuracy = {round(100 * accuracy, 2)}')

        ensemble_metrics_table = PrettyTable(["Precision", "Recall", "F1_Score", "Accuracy"])
        ensemble_metrics_table.add_row([round(100 * precision, 2), round(100 * recall, 2), round(100 * f1_score, 2), round(100 * accuracy, 2)])

        with open(os.path.join(destination_path, f"metrics_{model_name}.txt"), "w+") as txt_out:
            txt_out.write(str(ensemble_metrics_table))

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

    elif ensemble_method in ['wt_avg']:

        # we will construct a single (wt avg) model for each architecture type:
        wt_avg_model_types = list(set(model_types))
        wt_avg_tokenizers = []
        wt_avg_test_loaders = []
        wt_avg_constituent_models = []
        wt_avg_constituent_model_names = []

        # Loop through each architecture:
        for mt in wt_avg_model_types:

            idx_same_type = np.where(np.array(model_types) == mt)[0]

            # Pick a tokenizer/test loader (these shouldn't vary across models of same type)
            wt_avg_tokenizers.append(tokenizers[idx_same_type[0]])
            wt_avg_test_loaders.append(test_loaders[idx_same_type[0]])
            wt_avg_constituent_model_names.append(f"wt_avg_{mt}")

            # Consider the set of constituent models with this architecture
            models_to_avg = list(np.array(constituent_models)[idx_same_type]).copy()
            num_models = len(models_to_avg)
            scale_factor = 1. / num_models
            wt_dicts = [dict(m.named_parameters()) for m in models_to_avg]

            # begin with taking the weights from a single model
            avg_wt_dict = wt_dicts[0].copy()

            # for every weight matrix:
            for wt_name in avg_wt_dict.keys():

                # initialise with scaled weights from first model
                avg_wt_dict[wt_name].data.copy_(scale_factor * wt_dicts[0][wt_name].data)

                # add scaled weights from every other model
                for wt in wt_dicts[1:]:
                    avg_wt_dict[wt_name].data.copy_(avg_wt_dict[wt_name].data + scale_factor * wt[wt_name].data)

            # finalise the model weights
            models_to_avg[0].load_state_dict(avg_wt_dict, strict=False)
            wt_avg_constituent_models.append(models_to_avg[0])


            logging.info(f"Finished constructing wt_avg model for {mt}")
            
        logging.info(f"Finished constructing all required wt_avg models, now sending to second stage (majority voting)")
        # then send reulting set of models to evaluate_ensemble('majority')
        evaluate_ensemble(constituent_models = wt_avg_constituent_models,
                          constituent_model_names = wt_avg_constituent_model_names,
                          test_loaders = wt_avg_test_loaders,
                          destination_path = destination_path,
                          model_name = "ensemble_model",
                          tokenizers = wt_avg_tokenizers,
                          model_types = wt_avg_model_types,
                          test_file_path = test_file_path,
                          ensemble_method = 'majority')

    elif ensemble_method in ['latent']:

        ## TO DO: IMPLEMENT THIS METHOD ##
        # add to the 'wt_avg' code block when ready, method is similar
        raise NotImplementedError

def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))