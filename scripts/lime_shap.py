import shap
from transformers import pipeline
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer


def get_shap_scores(model, text, tokenizer):
    
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
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
        outputs = model(**tokenizer(input_string, return_tensors="pt", padding=True))
        tensor_logits = outputs[0]
        probas = F.softmax(tensor_logits, dim=1).detach().numpy()
        return probas

    lime_scores = []

    explainer = LimeTextExplainer(class_names=class_names)

    for t in text[1:]:
        tokens = [tok for tok in tokenizer.convert_ids_to_tokens(t) if tok not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]]
        t = " ".join(tokens)
        exp = explainer.explain_instance(t, predictor, num_features=len(tokens), num_samples=2000)
        exp_dict = exp.as_map()
        scores = ["_" for i in range(len(exp_dict[1]))]
        for s in exp_dict[1]:
            scores[s[0]] = s[1]

        lime_scores.append([0] + scores + [0])

    return lime_scores