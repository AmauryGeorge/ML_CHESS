import typing
import numpy as np
from itertools import groupby

def ctc_decoder(predictions: np.ndarray, chars: typing.Union[str, list]) -> typing.List[str]:
    """ CTC greedy decoder for predictions
    
    Args:
        predictions (np.ndarray): predictions from model
        chars (typing.Union[str, list]): list of characters

    Returns:
        typing.List[str]: list of words
    """
    
    # use argsort to find the index of the n highest probabilities
    argmax_preds = np.argsort(predictions, axis=-1)[:, :, :]

    grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds[:,:,-1]]
    # convert indexes to chars
    text = ["".join([chars[k] for k in group if k < len(chars)]) for group in grouped_preds]

    argmax_arr = argmax_preds[:,:,-1][0]
    values_at_argmax_arr = np.take_along_axis(predictions[0], argmax_arr[..., None], axis=-1)
    unsures = []
    c = list(chars)
    c.append("' '")
    for i in range(len(values_at_argmax_arr)):
        if np.exp(values_at_argmax_arr[i][0])< 0.9:
            most_likely_idx = np.argsort(predictions[0][i])[-2]
            idx = argmax_arr[i]
            new_preds = [i for i in argmax_preds[:,:,-1]]
            new_preds[0][i] = most_likely_idx
            new_grouped_preds = [[k for k,_ in groupby(preds)] for preds in new_preds]
            new_text = ["".join([chars[k] for k in group if k < len(chars)]) for group in new_grouped_preds]
            unsures.append(new_text)
    return text,unsures