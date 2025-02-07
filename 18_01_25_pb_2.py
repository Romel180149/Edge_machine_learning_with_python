
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np


x= np.array(['free money now ','call now to claim your prize','meet me at the park','lets catch up later','win a new car today','lunch plans?','who are you?','how are you?','whats pur number?']),
y= np.array([1,1,0,0,1,0,1,0,1,0])

# Data preparation Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

#data normalization
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

#training
clf = MultinomialNB()
clf.fit(x_train,y_train)

y_pre = clf.predict(x_test)
print(accuracy_score(y_test,y_pre))