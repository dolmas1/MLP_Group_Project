import torch


def get_attention_scores(attention_mask, attentions):
    layer_sum = torch.zeros(attentions[0].shape)

    for layer in attentions:
        layer_sum += layer

    layer_head_sum = torch.sum(layer_sum, 1)

    scores = [torch.masked_select(example[0], mask.bool()).tolist() for example, mask in zip(layer_head_sum, attention_mask)]

    return scores

