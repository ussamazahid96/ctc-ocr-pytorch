import math
import numpy as np

class seq_mnist_decoder():
    def __init__(self, labels, blank=0):
        self.blank_chr = blank
        self.labels = labels
    
    def decode(self, predictions, output_len, label_len):
        predictions = predictions.detach().numpy()
        output = []
        for i in range(output_len):
            pred = np.argmax(predictions[i, :])
            if (pred != self.blank_chr) and (pred != np.argmax(predictions[i-1, :])): # merging repeats:
                output.append(pred)
        return output

    def hit(self, pred, target):
        res = []
        for idx, word in enumerate(pred):
            res.append(word == target[idx])
        acc = np.mean(np.asarray(res))*100
        if math.isnan(acc):
            return 0.00
        else:
            return acc