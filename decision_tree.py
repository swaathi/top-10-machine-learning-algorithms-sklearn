from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = DecisionTreeClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nDecision Tree Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")

dot_file = 'visualizations/decision_tree.dot'
export_graphviz(model, out_file=dot_file, feature_names=Xtrain.columns.values)

# to convert .dot to .png
# dot -Tpng visualizations/decision_tree.dot -o visualizations/decision_tree.png
