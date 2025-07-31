import collections
def compute_f1_score(preds, labels):
    total_f1_score = 0
    for pred, truth in zip(preds, labels):
        pred_tokens = pred.split()
        truth_tokens = truth.split()
        
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            total_f1_score += 0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            total_f1_score += f1

    return total_f1_score/len(preds)