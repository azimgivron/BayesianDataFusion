from sklearn.metrics import roc_auc_score


def compute_auc_roc(Ytrue, scores):
    return roc_auc_score(Ytrue, scores)
