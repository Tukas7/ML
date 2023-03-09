import numpy as np
from sklearn import metrics


C="Cat"
F="Fish"
H="Hen"


y_true = [C,C,C,C,C,C, F,F,F,F,F,F,F,F,F,F, H,H,H,H,H,H,H,H,H]

y_pred = [C,C,C,C,H,F, C,C,C,C,C,C,H,H,F,F, C,C,C,H,H,H,H,H,H]

C = metrics.confusion_matrix(y_true, y_pred)
print(C)

print(metrics.classification_report(y_true, y_pred, digits=3))