from sklearn import svm
from sklearn import metrics
from base import Base

from sklearn.metrics import precision_recall_curve

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = svm.LinearSVC(random_state=800)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nSVM Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
print(metrics.classification_report(ypred, ytest))
