# import keras
#
# from keras.datasets import mnist
# import matplotlib.pyplot as plt
#
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# from skimage import feature
#
# dimx = dimy = 28
#
# fc, ft = feature.haar_like_feature_coord(dimx, dimy, ['type-2-x', 'type-2-y'])
# print(ft.shape)

# plot 4 images as gray scale
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# # show the plot
# plt.show()

#################### WARNING ###################################
# This code does not run correctly because presenting different
# altenatives
#################################################################


from skimage import feature
from skimage import transform
import sklearn
import numpy as np

# DATASET LOADING
from keras.datasets import mnist
from sklearn import ensemble
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#choose classes 4 and 8
x_train4 = x_train[y_train==4,:]
x_train8 = x_train[y_train==8,:]

def shuffledata(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

x_train = np.append(x_train4, x_train8, axis=0)
y_train = np.append(np.full(len(x_train4), -1), np.full(len(x_train8), 1))

(x_train,y_train) = shuffledata(x_train, y_train)

# HAAR FILTERS : version 1

feat_coord = np.array([list([[(0, 0), (0, 0)], [(0, 13), (0, 13)]]),
       list([[(0, 0), (13, 0)], [(0, 13), (13, 13)]]),
       list([[(0, 13), (0, 13)], [(0, 27), (0, 27)]]),
       list([[(0, 13), (13, 13)], [(0, 27), (13, 27)]]),
       list([[(13, 0), (13, 0)], [(13, 13), (13, 13)]]),
       list([[(13, 0), (27, 0)], [(13, 13), (27, 13)]]),
       list([[(13, 13), (13, 13)], [(13, 27), (13, 27)]]),
       list([[(13, 13), (27, 13)], [(13, 27), (27, 27)]]),
       list([[(27, 0), (27, 0)], [(27, 13), (27, 13)]]),
       list([[(27, 13), (27, 13)], [(27, 27), (27, 27)]]),
       list([[(0, 0), (0, 0)], [(13, 0), (13, 0)]]),
       list([[(0, 0), (0, 13)], [(13, 0), (13, 13)]]),
       list([[(0, 13), (0, 13)], [(13, 13), (13, 13)]]),
       list([[(0, 13), (0, 27)], [(13, 13), (13, 27)]]),
       list([[(0, 27), (0, 27)], [(13, 27), (13, 27)]]),
       list([[(13, 0), (13, 0)], [(27, 0), (27, 0)]]),
       list([[(13, 0), (13, 13)], [(27, 0), (27, 13)]]),
       list([[(13, 13), (13, 13)], [(27, 13), (27, 13)]]),
       list([[(13, 13), (13, 27)], [(27, 13), (27, 27)]]),
       list([[(13, 27), (13, 27)], [(27, 27), (27, 27)]])], dtype=object)

feat_type = np.array(['type-2-x', 'type-2-x', 'type-2-x', 'type-2-x', 'type-2-x', 'type-2-x', 'type-2-x', 'type-2-x', 'type-2-x', 'type-2-x', 'type-2-y', 'type-2-y', 'type-2-y', 'type-2-y', 'type-2-y',   'type-2-y', 'type-2-y', 'type-2-y', 'type-2-y', 'type-2-y'],    dtype=object)

# # HAAR FILTER : version 2
# radius = 30
# feat_coord, feat_type = feature.haar_like_feature_coord(28,28, ['type-2-x','type-2-y'])
# #reducing the number of filters
# i = 0
# while i < len(feat_coord):
#     if (feat_coord[i][0][1][0]-feat_coord[i][0][0][0])**2 + (feat_coord[i][0][1][1]-feat_coord[i][0][0][1])**2 < radius:
#         feat_coord = np.delete(feat_coord,i)
#         feat_type = np.delete(feat_type,i)
#     else:
#         i += 1
# # one over 4
# feat_coord = feat_coord[::4]
# feat_type = feat_type[::4]
print('features', feat_coord.shape)

first = True
for image in x_train:
    int_image = transform.integral_image(image)
    features = feature.haar_like_feature(int_image, 0, 0, 28, 28,feature_type=feat_type,feature_coord=feat_coord)
    if first:
        ftrain = [features]
    else:
        ftrain = np.append(ftrain,[features],axis=0)
    first = False

# TRAINING

myboosting = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
myboosting.fit(ftrain, y_train)

# TEST
# preparation of test data
#choose classes 4 and 8
x_test4 = x_test[y_test==4, :]
x_test8 = x_test[y_test==8, :]


x_test = np.append(x_test4, x_test8, axis=0)
y_test = np.append(np.full(len(x_test4),-1), np.full(len(x_test8),1))

(x_test,y_test) = shuffledata(x_test,y_test)

first = True
for image in x_test:
    int_image = transform.integral_image(image)
    features = feature.haar_like_feature(int_image, 0, 0, 28, 28,feature_type=feat_type,feature_coord=feat_coord)
    if first:
        ftest = [features]
    else:
        ftest = np.append(ftest,[features],axis=0)
    first = False

# evaluation
y_pred = myboosting.predict(ftest)
print('confusion matrix', confusion_matrix(y_test,y_pred))

