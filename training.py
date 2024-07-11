import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc


Data = pd.read_csv('data.csv')

X = Data["Email Text"].values
y = Data["Email Type"].values

from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# define the Classifier
classifier = Pipeline([("tfidf",TfidfVectorizer() ),("classifier",RandomForestClassifier(n_estimators=10))])

classifier.fit(X_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, columns=classifier.classes_, index=classifier.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

y_prob = classifier.predict_proba(x_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classifier.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_prob[:, i], pos_label=classifier.classes_[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute Precision-Recall curve and area for each class
precision = dict()
recall = dict()
pr_auc = dict()
for i in range(len(classifier.classes_)):
    precision[i], recall[i], _ = precision_recall_curve(y_test, y_prob[:, i], pos_label=classifier.classes_[i])
    pr_auc[i] = auc(recall[i], precision[i])

# Plotting ROC curve
plt.figure(figsize=(10, 5))
for i in range(len(classifier.classes_)):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], classifier.classes_[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plotting Precision-Recall curve
plt.figure(figsize=(10, 5))
for i in range(len(classifier.classes_)):
    plt.plot(recall[i], precision[i], label='Precision-Recall curve (area = %0.2f) for class %s' % (pr_auc[i], classifier.classes_[i]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

import joblib
joblib.dump(classifier,'eTc.sav')