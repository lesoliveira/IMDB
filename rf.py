#!/usr/bin/env python3

## caracteristicas geradas pelo BERT

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('imdb_master-UTF.csv' )
data = data.iloc[:-50000]
label = data['label'] ;
data['label'] = data['label'].map({'neg': 0, 'pos': 1})

# Convert the mapped column to a NumPy array
y_data = data['label'].to_numpy()
X_data = np.loadtxt('embeddings.txt')

# Generate a range of indices corresponding to X_data
indices = np.arange(len(X_data))

#X_train, X_test, y_train, y_test = train_test_split (X_data, y_data, test_size=0.2, random_state=5)
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
	X_data, y_data, indices, test_size=0.5, random_state=5
)
#for i in range(len(test_idx)):
#    print (test_idx[i], y_data[test_idx[i]])
#print ('Training the Random Forest')
#forest = DecisionTreeClassifier().fit(X_train, y_train)
#forest = RandomForestClassifier(n_estimators=150).fit(X_train, y_train)
forest = LogisticRegression().fit(X_train, y_train)
y_pred = forest.predict(X_test)

print('RF score = {}'.format(forest.score(X_test, y_test)))
cm = confusion_matrix(y_test, y_pred)
print (cm)
print (classification_report(y_test, y_pred ))

#forest = AdaBoostClassifier(n_estimators=150).fit(X_train, y_train)
#print('RF2 score = {}'.format(forest.score(X_test, y_test)))

