import math
import numpy as np

class seq_mnist_decoder():
    def __init__(self, labels, blank=0):
        self.blank_chr = blank
        self.labels = labels
    
    def decode(self, predictions, output_len, label_len):
        predictions = predictions.data.cpu().numpy()
        output = []
        for i in range(output_len):
            pred = np.argmax(predictions[i, :])
            if (pred != self.blank_chr) and (pred != np.argmax(predictions[i-1, :])): # merging repeats and removing blank character (0)
                output.append(pred-1)
        return np.asarray(output)

    def hit(self, pred, target):
        res = []
        for idx, word in enumerate(target):
            if idx < len(pred):
                item = pred[idx]
            else:
                item = 10
            res.append(word == item)
        acc = np.mean(np.asarray(res))*100
        if math.isnan(acc):
            return 0.00
        else:
            return acc

    def to_string(self, in_str):
        out_str = ''
        for i in range(in_str.shape[0]):
            out_str += str(in_str[i])
        return out_str