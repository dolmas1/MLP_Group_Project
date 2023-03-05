import shap
from transformers import pipeline
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_shap_scores(model, text, tokenizer):
    
    pipe = pipeline("text-classification", model=model.to(device), tokenizer=tokenizer, device=device)
    explainer = shap.Explainer(pipe)

    sentences = []
    for t in text:
        t = tokenizer.decode(t).replace(tokenizer.pad_token, "").replace(tokenizer.cls_token, "").replace(tokenizer.sep_token, "")
        sentences.append(t)

    final_values = [list(val[:,1]) for val in explainer(sentences).values] # :,1 because only want scores for positive class
    
    return final_values

def get_lime_scores(model, text, tokenizer):

    class_names = ['negative', 'positive']

    def predictor(input_string):

        new_input_string = []
        for i in input_string:
            s = i.replace("Ġ", "").replace("##", "")
            new_input_string.append(s)

        outputs = model(**tokenizer(new_input_string, return_tensors="pt", padding=True).to(device))
        tensor_logits = outputs[0]
        probas = F.softmax(tensor_logits, dim=1).detach().cpu().numpy()
        return probas

    lime_scores = []

    explainer = LimeTextExplainer(class_names=class_names)

    for t in text:
        tokens = [tok for tok in tokenizer.convert_ids_to_tokens(t) if tok not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]]
        t = " ".join(tokens)
        exp = explainer.explain_instance(t, predictor, num_features=len(tokens), num_samples=100)
        exp_dict = exp.as_map()
        scores = ["_" for i in range(len(exp_dict[1]))]
        for s in exp_dict[1]:
            scores[s[0]] = s[1]

        lime_scores.append([0] + scores + [0])


    return lime_scores