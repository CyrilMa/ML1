from sklearn import decomposition
from sklearn import neural_network
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import gini

### Regressor
Regressor = {}
Regressor["AdaBoost"] = ensemble.AdaBoostRegressor()
Regressor["GradBoost"] = ensemble.GradientBoostingRegressor()
Regressor["RandomForest"] = ensemble.RandomForestRegressor()
Regressor["NeuralNetwork"] = neural_network.MLPRegressor()

### Classifier
Classifier = {}
Classifier["AdaBoost"] = ensemble.AdaBoostClassifier()
Classifier["GradBoost"] = ensemble.GradientBoostingClassifier()
Classifier["RandomForest"] = ensemble.RandomForestClassifier()
Classifier["NeuralNetwork"] = neural_network.MLPClassifier()

def Regressors(X, Y):
	X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y, test_size=0.2, random_state=None)
	for key, regressor in Regressor.items():
		regressor.fit(X_train, Y_train)
		Y_pred = regressor.predict(X_test)
		print('By {}'.format(key))
		gini.gini_visualization(Y_test, Y_pred, True)
		print("------------------------------------")

def Classifiers(X, Y):
	X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y, test_size=0.2, random_state=None)
	for key, classifier in Classifier.items():
		classifier.fit(X_train, Y_train)
		Y_pred = classifier.predict(X_test)
		print('By {}'.format(key))
		print(accuracy_score(Y_test, Y_pred))
		print("------------------------------------")
