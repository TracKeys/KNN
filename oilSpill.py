import os, numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

input_dir = r'C:\Users\user\dataset\test'
catgories = ['Oil-Spill', 'Non-Oil-Spill']

data   = []
labels = []  #   suppervised ML --> data labled

for index, catgorie in enumerate(catgories):
    for file in os.listdir(os.path.join(input_dir,catgorie )):
        img_path = os.path.join(input_dir,catgorie,file)
        image = imread(img_path)
        image = resize(image, (15,15))
        data.append(image.flatten())
        labels.append(index)
    print('Done',catgorie)

data = np.asarray(data)
labels = np.asarray(labels)
# unseen data

X_train, X_test, y_train,y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify= labels)
print('Split')
clf = SVC()
# tunining 
parameters = [{'gamma':[0.01,0.001,0.00001], 'C':[1, 10,100,1000]}]
# It will train the model 12 times to achive the best margin
grid_search = GridSearchCV(clf, parameters)
print('train....!')
grid_search.fit(X_train, y_train)

#print(grid_search.best_estimator_)
best_estimator = grid_search.best_estimator_

y_predict = best_estimator.predict(X_test)

score = accuracy_score(y_predict,  y_test)
pickle.dump(best_estimator, open('./Oil_BestEstimator_Model.p', 'wb'))
print('{}% of samples were correctly classified'.format(str(score * 100)))























