from sklearn import svm
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = svm.LinearSVC(random_state=800)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nSVM Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
