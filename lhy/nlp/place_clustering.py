from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import data
import csv
import pandas as pd

place_w2v = data.read_pickle('agency_2_vec_origin.pickle')
token = list(place_w2v.keys())

min_k = 5
max_k = 6

K = range(min_k, max_k + 1)
mean_distortions = []
sc_scores = []

w2v = []
for key,value in place_w2v.items():
    w2v.append(value)

for k in K:
    kmeans_sk = KMeans(n_clusters=k, max_iter=600, tol = 1e-6).fit(w2v)
    sc_score = silhouette_score(w2v, kmeans_sk.labels_, metric='euclidean')
    sc_scores.append(sc_score)
# print(sc_scores)

bestK = np.argmax(sc_scores) + min_k
# print(bestK)
km_2d = KMeans(n_clusters=bestK, algorithm="full").fit(w2v)

cluster_result = []

for i in range(0,bestK):
    temp = []
    cluster_result.append(temp)

for i in range(0, len(place_w2v)):
    cluster_index = (km_2d.labels_)[i]
    cluster_result[cluster_index].append(token[i])
# print(km_2d.labels_)
# print(bestK)
# print(cluster_result)

pca = PCA(n_components=2)
w2v_2d = pca.fit_transform(w2v)

# 建立一个字典储存聚类标记

dict_cluster_label = {}
dict_place_label = {}
for label, item in enumerate(cluster_result):
    dict_cluster_label[label] = item
    for i in item:
        dict_place_label[i] = label

f = open('原始数据行业等级label类字典.txt', 'w')
for key,value in dict_place_label.items():
    f.write('{},{}'.format(key, value))
    f.write('\n')
f.close()


f = open('原始数据聚类标记的行业等级等类字典.txt', 'w')
for key,value in dict_cluster_label.items():
    f.write('-----这是第{}类-----'.format(key))
    f.write('\n')
    for i in value:
        f.write(i + '\n')
f.close()

# #storage
place_cluster_obj = open('原始处置单位聚类.txt', 'a')
place_cluster_obj.write('\n')
place_cluster_obj.write('======Best K is {}======'.format(bestK))
place_cluster_obj.write('\n')
for i in cluster_result:
    place_cluster_obj.write(str(i))
    place_cluster_obj.write('\n')
place_cluster_obj.write('')
place_cluster_obj.write('\n')
place_cluster_obj.write('======Best K is {}======'.format(bestK))
place_cluster_obj.write('\n')
place_cluster_obj.close()
#