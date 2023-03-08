import shap
from transformers import pipeline
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_shap_scores(model, text, tokenizer, original_text):
    
    pipe = pipeline("text-classification", model=model.to(device), tokenizer=tokenizer, device=device)
    explainer = shap.Explainer(pipe)

    final_values = []    
    for val, t in zip(explainer(original_text).values, text):
        l = len([x for x in t if x != tokenizer.pad_token_id])
        scores = list(val[:,1]) # :,1 because only want scores for positive class
        assert l == len(scores) # check that there are the right amount of scores, one for each token
        final_values.append(scores)

    return final_values

def get_lime_scores(model, text, tokenizer):

    class_names = ['negative', 'positive']

    def indices(lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    def predictor(input_string):

        padded_input_ids = []
        for index, i in enumerate(input_string): 
            if index != 0:
                i = i.replace(tokenizer.cls_token, "").replace(tokenizer.sep_token, "") # remove SOS and EOS if it is the middle because of perturbation
                i = tokenizer.cls_token + " " + i + " " + tokenizer.sep_token # append SOS and EOS
            
            ids = tokenizer.convert_tokens_to_ids(i.split()) # convert it to ids
            padded_ids = ids + [tokenizer.pad_token_id for _ in range(512 - len(ids))] # pad the ids
            assert len(padded_ids) == 512
            padded_input_ids.append(padded_ids)

        outputs = model(torch.tensor(padded_input_ids).to(device), shap=True)
        tensor_logits = outputs[0]
        probas = F.softmax(tensor_logits, dim=1).detach().cpu().numpy()
        return probas

    lime_scores = []

    explainer = LimeTextExplainer(class_names=class_names, split_expression="\s")

    for t in text:
        ref_id = [x.item() for x in t if x != tokenizer.pad_token_id]

        tokens = [tok for tok in tokenizer.convert_ids_to_tokens(t) if tok not in [tokenizer.pad_token]]

        double_tokens = dict((x, indices(tokens, x)) for x in set(tokens) if tokens.count(x) > 1)

        t = " ".join(tokens)
        exp = explainer.explain_instance(t, predictor, num_features=len(ref_id), num_samples=100)
        
        scores = []
        for index, tok in enumerate(tokens): 
            for (word, score) in exp.as_list():
                if word == tok:
                    if word in double_tokens:
                        number_of_duplicates = len(double_tokens[word])
                        lime_score = score / number_of_duplicates
                        scores.append(lime_score)
                    else:
                        scores.append(score)


        assert len(scores) == len(ref_id)

        lime_scores.append(scores)
    return lime_scores