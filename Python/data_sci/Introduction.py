from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#[height, weight, shoe size]
x = [
    [181, 80, 44], 
    [177, 70, 43], 
    [160, 60, 38], 
    [154, 54, 37], 
    [166, 65, 40],
    [190, 90, 47], 
    [175, 64, 39],
    [177, 70, 40], 
    [159, 55, 37], 
    [171, 75, 42], 
    [181, 85, 43]
    ]

y = ['male', 
     'male', 
     'female', 
     'female', 
     'male', 
     'male', 
     'female', 
     'female',
     'female', 
     'male', 
     'male']

#check the input count
assert len(x) == len(y) , "Data count doens't match"

'''Decision Tree model'''
decision_tree = tree.DecisionTreeClassifier()
dt_model = decision_tree.fit(x,y)

'''Random Forest model'''
random_forest = RandomForestClassifier()
rf_model = random_forest.fit(x,y)

'''SVM model'''
svm = SVC()
svm_model = svm.fit(x,y)

'''K-Nearest Neighbors'''
knn = KNeighborsClassifier()
knn_model = knn.fit(x,y)


data_to_predict = [190,70,43]
'''prediction output'''
dt_predict = dt_model.predict([data_to_predict])
rf_predict = rf_model.predict([data_to_predict])
svm_predict = svm_model.predict([data_to_predict])
knn_predict = knn_model.predict([data_to_predict])

print("Input: ", data_to_predict)
print("Decision Tree: ",dt_predict)
print("Random Forest: ",rf_predict)
print("SVM: ",svm_predict)
print("KNN: ",knn_predict)
