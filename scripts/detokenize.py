import re
import numpy as np

def detokenize_single(toks, inter, model_type):
    """Function to perform first step of detokenization for a single set of test sentence tokens/interpretability scores

    Args:
        toks (list): list of tokens, the output of running a model tokenizer on raw input sentence string
        inter (np.array): np array of dimension (# interpretability measures, # tokens)
                          containing the range of different local interpretability scores for each token
        model_type (str): must contain 'bert' or 'roberta', controls the type of detokenization to apply.
        
    Outputs:
        merged_toks_op (list): list of 'merged tokens', more closely approximating one token per word.
                               Tokens containing 'Ġ' or '##' have been concatenated, along with individual puctuation tokens.
        merged_inter_op (np.array): numpy array containing interpretability scores for each 'merged token'.
                                    Where tokens have been merged, their corresponding interpretability scores are summed.
    """
    
    toks = toks.copy()
    
    # First, deal with the model-specific tokenisation:
    if 'roberta' in model_type:
        
        toks[0] = '<SOS>'
        toks[-1] = 'Ġ<EOS>'
        
        merged_toks = [s.lower() for s in toks[:2]]
        merged_inter = inter[:, :2]
        inter = np.delete(inter, [0,1], 1)
        
        for tok in toks[2:]:
            # Tokens starting 'Ġ' are new words, otherwise merge with previous token
            if tok[:1] == 'Ġ':
                merged_toks.append(tok[1:].lower())
                merged_inter = np.append(merged_inter, inter[:, 0].reshape((-1,1)), axis=1)
            else:
                merged_toks[-1] += tok.lower()
                
                # Add the interpretability scores together
                merged_inter[:,-1] = merged_inter[:,-1] + inter[:, 0]
            
            inter = np.delete(inter, 0, 1)
            
    elif 'bert' in model_type:
        
        toks[0] = '<SOS>'
        toks[-1] = '<EOS>'
        
        merged_toks = [s.lower() for s in toks[:1]]
        merged_inter = inter[:, 0].reshape(-1, 1)
        inter = np.delete(inter, 0, 1)
        
        for tok in toks[1:]:
            # Tokens starting '##' are merged with previous token
            if tok[:2] == '##':
                merged_toks[-1] += tok[2:].lower()
                
                # Add the interpretability scores together
                merged_inter[:,-1] = merged_inter[:,-1] + inter[:, 0]
                
            else:
                merged_toks.append(tok.lower())
                merged_inter = np.append(merged_inter, inter[:, 0].reshape((-1,1)), axis=1)
                
            inter = np.delete(inter, 0, 1)


    # Next, merge isolated punctuation characters with the preceeding token:
    merged_toks_op = []
    merged_inter_op = np.zeros((inter.shape[0],0))

    for tok in merged_toks:
        if re.match('^[^\w\s]+$', tok) is not None:
            merged_toks_op[-1] += tok
            
            # Add the interpretability scores together
            merged_inter_op[:,-1] = merged_inter_op[:,-1] + merged_inter[:, 0]
            
        else:
            merged_toks_op.append(tok)
            merged_inter_op = np.append(merged_inter_op, merged_inter[:, 0].reshape((-1,1)), axis=1)
            
        merged_inter = np.delete(merged_inter, 0, 1)
    
    assert len(merged_toks_op) == merged_inter_op.shape[1], "error in detokenization, please debug"
    
    return merged_toks_op, merged_inter_op


def force_agreement(toks_a_input, toks_b_input, inter_a, inter_b):
    """Function to perform second step of detokenization, across a pair of (merged_token, merged_interpretability) output tuples from two different models

    Args:
        toks_a_input (list): list of merged tokens for the first model, the output of running detokenize() on a set of tokens
        toks_b_input (list): list of merged tokens for the second model, the output of running detokenize() on a set of tokens
        inter_a (np.array): np array of dimension (# interpretability measures, # merged tokens)
                            containing the range of different local interpretability scores for each merged token from FIRST model
        inter_b (np.array): np array of dimension (# interpretability measures, # merged tokens)
                            containing the range of different local interpretability scores for each merged token from SECOND model
        
    Outputs:
        unified_toks (list): list of fully unified tokens, where tokens from different models have been matched and combined wherever possible
                             (note that a small number of tokens/associated interpretability scores may be discarded from each model, where agreement could not be made)
        merged_inter_a (np.array): numpy array containing interpretability scores for each fully unified token for model a.
                                   Where tokens have been merged, their corresponding interpretability scores are summed.
        merged_inter_b (np.array): numpy array containing interpretability scores for each fully unified token for model b.
                                   Where tokens have been merged, their corresponding interpretability scores are summed.
    """
    
    toks_a = toks_a_input.copy()
    toks_b = toks_b_input.copy()
    inter_a = inter_a.copy()
    inter_b = inter_b.copy()
    
    unified_toks = []
    
    inter_a_dim = inter_a.shape[0]
    inter_b_dim = inter_b.shape[0]
    
    merged_inter_a = np.zeros((inter_a_dim, 0))
    merged_inter_b = np.zeros((inter_b_dim, 0))

    
    
    while len(toks_a) + len(toks_b) > 0:
        
        
        if len(toks_a) > 0 and len(toks_b) > 0:
            
            # where there is a direct match, copy to output and continue
            if toks_a[0].lower() == toks_b[0].lower():
                unified_toks.append(toks_a.pop(0))
                toks_b.pop(0)
                
                merged_inter_a = np.append(merged_inter_a, inter_a[:, 0].reshape((-1,1)), axis=1)
                inter_a = np.delete(inter_a, 0, 1)
                merged_inter_b = np.append(merged_inter_b, inter_b[:, 0].reshape((-1,1)), axis=1)
                inter_b = np.delete(inter_b, 0, 1)
        

            # case where multiple tokens in both lists:
            elif len(toks_a) > 1 and len(toks_b) > 1:
                    
                # where token for model_b matches concatenation of next two tokens for model_a, merge for model_a
                if toks_a[0].lower() + toks_a[1].lower() == toks_b[0].lower():
                    
                    unified_toks.append(toks_b.pop(0))
                    toks_a.pop(0)
                    toks_a.pop(0)

                    merged_inter_b = np.append(merged_inter_b, inter_b[:, 0].reshape((-1,1)), axis=1)
                    inter_b = np.delete(inter_b, 0, 1)
                    merged_inter_a = np.append(merged_inter_a, inter_a[:, 0:2].sum(axis=1).reshape((-1,1)), axis=1)
                    inter_a = np.delete(inter_a, [0, 1], 1)
                    
                # when model_b token does not match model_a token, but DOES match the NEXT model_a token, throw away model_a token
                elif toks_b[0].lower() == toks_a[1].lower():
                    toks_a.pop(0)
                    
                    inter_a = np.delete(inter_a, 0, 1)

                # where token for model_a matches concatenation of next two tokens for model_b, merge for model_b
                elif toks_b[0].lower() + toks_b[1].lower() == toks_a[0].lower():
                    unified_toks.append(toks_a.pop(0))
                    toks_b.pop(0)
                    toks_b.pop(0)
                    
                    merged_inter_a = np.append(merged_inter_a, inter_a[:, 0].reshape((-1,1)), axis=1)
                    inter_a = np.delete(inter_a, 0, 1)
                    merged_inter_b = np.append(merged_inter_b, inter_b[:, 0:2].sum(axis=1).reshape((-1,1)), axis=1)
                    inter_b = np.delete(inter_b, [0, 1], 1)
                        
                # when model_a token does not match model_b token, but DOES match the NEXT model_b token, throw away model_b token
                elif toks_a[0].lower() == toks_b[1].lower():
                    toks_b.pop(0)
                    
                    inter_b = np.delete(inter_b, 0, 1)
                
                # when the next two model_a tokens match the next two model_b tokens, merge them both
                elif toks_a[0].lower() + toks_a[1].lower() == toks_b[0].lower() + toks_b[1].lower():
                    unified_toks.append(toks_a.pop(0) + toks_a.pop(0))
                    toks_b.pop(0)
                    toks_b.pop(0)
                    
                    merged_inter_a = np.append(merged_inter_a, inter_a[:, 0:2].sum(axis=1).reshape((-1,1)), axis=1)
                    inter_a = np.delete(inter_a, [0, 1], 1)
                    merged_inter_b = np.append(merged_inter_b, inter_b[:, 0:2].sum(axis=1).reshape((-1,1)), axis=1)
                    inter_b = np.delete(inter_b, [0, 1], 1)
                
                # otherwise, throw both away
                else:
                    toks_a.pop(0)
                    toks_b.pop(0)
                    
                    inter_a = np.delete(inter_a, 0, 1)
                    inter_b = np.delete(inter_b, 0, 1)

                
            # case where only single token left in b:
            elif len(toks_b) == 1:
                    
                    # where token for model_b matches concatenation of next two tokens for model_a, merge for model_a
                    if toks_a[0].lower() + toks_a[1].lower() == toks_b[0].lower():
                        unified_toks.append(toks_b.pop(0))
                        toks_a.pop(0)
                        toks_a.pop(0)
                        
                        merged_inter_b = np.append(merged_inter_b, inter_b[:, 0].reshape((-1,1)), axis=1)
                        inter_b = np.delete(inter_b, 0, 1)
                        merged_inter_a = np.append(merged_inter_a, inter_a[:, 0:2].sum(axis=1).reshape((-1,1)), axis=1)
                        inter_a = np.delete(inter_a, [0, 1], 1)
                    
                    # otherwise, throw away a
                    else:
                        toks_a.pop(0)
                        
                        inter_a = np.delete(inter_a, 0, 1)

            # case where only single token left in a:
            elif len(toks_a) == 1:
                    
                    # where token for model_a matches concatenation of next two tokens for model_b, merge for model_b
                    if toks_b[0].lower() + toks_b[1].lower() == toks_a[0].lower():
                        unified_toks.append(toks_a.pop(0))
                        toks_b.pop(0)
                        toks_b.pop(0)
                        
                        merged_inter_a = np.append(merged_inter_a, inter_a[:, 0].reshape((-1,1)), axis=1)
                        inter_a = np.delete(inter_a, 0, 1)
                        merged_inter_b = np.append(merged_inter_b, inter_b[:, 0:2].sum(axis=1).reshape((-1,1)), axis=1)
                        inter_b = np.delete(inter_b, [0, 1], 1)
                        
                    # otherwise, throw away b
                    else:
                        toks_b.pop(0)
                        
                        inter_b = np.delete(inter_b, 0, 1)


        # throw away a if exists
        elif len(toks_a) > 0:
            toks_a.pop(0)
            
            inter_a = np.delete(inter_a, 0, 1)
            
        # throw away b if exists
        elif len(toks_b) > 0:
            toks_b.pop(0)
            
            inter_b = np.delete(inter_b, 0, 1)
    
    assert merged_inter_a.shape == (inter_a_dim, len(unified_toks)), "error in detokenization, please debug"
    assert merged_inter_b.shape == (inter_b_dim, len(unified_toks)), "error in detokenization, please debug"

    return unified_toks, merged_inter_a, merged_inter_b