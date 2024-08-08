import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ARCMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.true_labels = []
        self.pred_labels = []
    
    def update(self, outputs, targets):
        pred = outputs.argmax(dim=1).cpu().numpy()
        true = targets.cpu().numpy()
        self.true_labels.extend(true)
        self.pred_labels.extend(pred)
    
    def compute(self):
        return {
            'accuracy': accuracy_score(self.true_labels, self.pred_labels),
            'precision': precision_score(self.true_labels, self.pred_labels, average='weighted'),
            'recall': recall_score(self.true_labels, self.pred_labels, average='weighted'),
            'f1': f1_score(self.true_labels, self.pred_labels, average='weighted')
        }