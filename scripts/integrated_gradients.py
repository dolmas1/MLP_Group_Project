import torch
from captum.attr import LayerIntegratedGradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_integrated_gradients_score(text, labels, tokens, tokenizer, model, model_name):

    attributions_list = []
    def model_output(inputs):
        m_output = model(inputs, lig=True)
        return m_output[0]
    if "roberta" in model_name:
        lig = LayerIntegratedGradients(model_output, model.encoder.roberta.embeddings)

    elif "bert" in model_name:
        lig = LayerIntegratedGradients(model_output, model.encoder.bert.embeddings)
    
    else:
        raise Exception(f"Don't know the model type {model_name}!")

    for toks, input_ids, l in zip(tokens, text, labels):
        
        original_input_ids, baseline_input_ids = construct_input_and_baseline(tokenizer, input_ids, toks)

        attributions, delta = lig.attribute(inputs= original_input_ids, baselines= baseline_input_ids, target=l,
                        return_convergence_delta=True, internal_batch_size=1)


        attributions_sum = attributions.sum(dim=-1).squeeze(0)
        normed_attributions_sum = attributions_sum / torch.norm(attributions_sum)
        attributions_list.append(normed_attributions_sum)

    return attributions_list

def construct_input_and_baseline(tokenizer, input_ids, toks):
    baseline = [tokenizer.cls_token_id] +  [tokenizer.pad_token_id] * (len(toks) -2) + [tokenizer.sep_token_id]
    input_ids = [i for i in input_ids if i != tokenizer.pad_token_id]
    assert len(baseline) == len(input_ids)    
    return torch.tensor([input_ids], device=device), torch.tensor([baseline], device=device)
 
