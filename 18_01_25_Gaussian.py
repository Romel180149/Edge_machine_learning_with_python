import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("CC GENERAL.csv")
print(df.head())
print(df.info())

# Feature and target selection
x = df.iloc[:, [1, 4, 7, 9]].values
y = df.iloc[:, 11].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Standard scaling
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

# Initialize models
svc_classifier = SVC(kernel='rbf', random_state=0, verbose=False)
dt_classifier = DecisionTreeClassifier(random_state=0)
k_classifier = KNeighborsClassifier(n_neighbors=3)
gnb_classifier = GaussianNB()
mnb_classifier = MultinomialNB()  # Multinomial Naive Bayes

# Train models and evaluate
# SVC
svc_classifier.fit(x_train, y_train)
svc_y_pred = svc_classifier.predict(x_test)
svc_acc = accuracy_score(y_test, svc_y_pred)
svc_f1 = f1_score(y_test, svc_y_pred, average='weighted')
svc_recall_sc = recall_score(y_test, svc_y_pred, average='weighted')
svc_pre_sc = precision_score(y_test, svc_y_pred, average='weighted')

# Decision Tree
dt_classifier.fit(x_train, y_train)
dt_y_pred = dt_classifier.predict(x_test)
dt_acc = accuracy_score(y_test, dt_y_pred)
dt_f1 = f1_score(y_test, dt_y_pred, average='weighted')
dt_recall_sc = recall_score(y_test, dt_y_pred, average='weighted')
dt_pre_sc = precision_score(y_test, dt_y_pred, average='weighted')

# KNN
k_classifier.fit(x_train, y_train)
k_y_pred = k_classifier.predict(x_test)
k_acc = accuracy_score(y_test, k_y_pred)
k_f1 = f1_score(y_test, k_y_pred, average='weighted')
k_recall_sc = recall_score(y_test, k_y_pred, average='weighted')
k_pre_sc = precision_score(y_test, k_y_pred, average='weighted')

# Gaussian Naive Bayes
gnb_classifier.fit(x_train, y_train)
gnb_y_pred = gnb_classifier.predict(x_test)
gnb_acc = accuracy_score(y_test, gnb_y_pred)
gnb_f1 = f1_score(y_test, gnb_y_pred, average='weighted')
gnb_recall_sc = recall_score(y_test, gnb_y_pred, average='weighted')
gnb_pre_sc = precision_score(y_test, gnb_y_pred, average='weighted')

# Multinomial Naive Bayes
# MultinomialNB does not work well with scaled data. Use raw counts instead.
x_train_mnb = df.iloc[:, [1, 4, 7, 9]].values
x_test_mnb = df.iloc[:, [1, 4, 7, 9]].values
x_train_mnb, x_test_mnb, y_train_mnb, y_test_mnb = train_test_split(
    x_train_mnb, y, test_size=0.20, random_state=0
)
mnb_classifier.fit(x_train_mnb, y_train_mnb)
mnb_y_pred = mnb_classifier.predict(x_test_mnb)
mnb_acc = accuracy_score(y_test_mnb, mnb_y_pred)
mnb_f1 = f1_score(y_test_mnb, mnb_y_pred, average='weighted')
mnb_recall_sc = recall_score(y_test_mnb, mnb_y_pred, average='weighted')
mnb_pre_sc = precision_score(y_test_mnb, mnb_y_pred, average='weighted')

# Print results
print("SVC:", [svc_acc, svc_f1, svc_recall_sc, svc_pre_sc])
print("Decision Tree:", [dt_acc, dt_f1, dt_recall_sc, dt_pre_sc])
print("KNN:", [k_acc, k_f1, k_recall_sc, k_pre_sc])
print("GaussianNB:", [gnb_acc, gnb_f1, gnb_recall_sc, gnb_pre_sc])
print("MultinomialNB:", [mnb_acc, mnb_f1, mnb_recall_sc, mnb_pre_sc])

# Plot performance comparison
metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
data = np.array([[svc_acc, svc_f1, svc_recall_sc, svc_pre_sc],
                 [dt_acc, dt_f1, dt_recall_sc, dt_pre_sc],
                 [k_acc, k_f1, k_recall_sc, k_pre_sc],
                 [gnb_acc, gnb_f1, gnb_recall_sc, gnb_pre_sc],
                 [mnb_acc, mnb_f1, mnb_recall_sc, mnb_pre_sc]])

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(metrics))
width = 0.15

ax.bar(x - 2 * width, data[0], width, label='SVC')
ax.bar(x - width, data[1], width, label='Decision Tree')
ax.bar(x, data[2], width, label='KNN')
ax.bar(x + width, data[3], width, label='GaussianNB')
ax.bar(x + 2 * width, data[4], width, label='MultinomialNB')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Algorithm Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()
