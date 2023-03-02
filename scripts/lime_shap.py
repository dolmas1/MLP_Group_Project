import shap
from transformers import pipeline


def get_shap_scores(model, text, tokenizer):
    
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    explainer = shap.Explainer(pipe) 

    sentences = []
    for t in text:
        t = tokenizer.decode(t).replace(tokenizer.pad_token, "").replace(tokenizer.cls_token, "").replace(tokenizer.sep_token, "")
        sentences.append(t)

    final_values = [list(val[:,1]) for val in explainer(sentences).values] # :,1 because only want scores for positive class
    
    return final_values