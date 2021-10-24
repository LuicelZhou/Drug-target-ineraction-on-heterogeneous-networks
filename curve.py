import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from scipy import interp
from sklearn.metrics import roc_auc_score

line_color = ['magenta', 'darkblue', 'green', 'saddlebrown', 'crimson']

y_true = []
y_scores = []
tprs = []
aucs = []
precisions = []
auprs = []
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
plt.figure(figsize=(10, 10))

for i in range(5):
    label = np.loadtxt('label' + str(i))
    y_true = np.append(y_true, label)
    pred = np.loadtxt('pred' + str(i))
    y_scores = np.append(y_scores, pred)

    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = roc_auc_score(label, pred)
    aucs.append(roc_auc)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    #plt.plot(fpr, tpr, lw=1.5, alpha=0.5, color=line_color[i], ls='--',
             #label='ROC fold %d (AUC = %0.3f)' % (i + 1, roc_auc))

    recall, precision, thresholds = precision_recall_curve(label, pred)
    aupr = average_precision_score(label, pred)
    auprs.append(aupr)
    precisions.append(interp(mean_recall, recall, precision))
    precisions[-1][0] = 0.0
    #注释
    plt.plot(precision, recall, lw=1.5, alpha=0.5, color=line_color[i], ls='--',
              label='PR fold %d (AUPR = %0.3f)' % (i+1, aupr))
    #注释

#mean_tpr = np.mean(tprs, axis=0)
#mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
#plt.plot(mean_fpr, mean_tpr, color='darkorange',
         #label=r'Mean ROC (AUC = %.3f)' % mean_auc,
         #lw=2.5, alpha=.8)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend(loc="lower right")

#注释
me_recall, me_precision, thre = precision_recall_curve(y_true, y_scores)
mean_precision = np.mean(precisions, axis=0)
mean_precision[-1] = 0.0
plt.plot(mean_precision, mean_recall, color='darkorange',
          label=r'Mean PR (AUPR = 0.967)',
          lw=2.5, alpha=.8)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
#注释
#plt.xlim([-0.05, 1.05])
#plt.ylim([0.4, 1.05])
#注释
plt.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
#注释
plt.show()
