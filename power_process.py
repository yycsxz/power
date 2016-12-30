#coding:utf-8
import sys
import pandas as pd
import numpy as np

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import matplotlib


def draw_line(lines, cut = 1):
	import matplotlib.pyplot as plt
	# lines=[[0,1,2,3,4],[1,1,1,1,1],[4,4,4,4,4]]
	# lines = np.array(lines)
	lines = np.transpose(lines)
	plt.plot(lines)
	#plt.ylabel('slope_' + str(cut))
	plt.show()
def cal_slope(df, cut=1):
	#print df[47]
	slope_df = pd.DataFrame(columns = [i for i in range(int(POINT_NUM/cut)-1)])
	for i in range(int(POINT_NUM/cut) - 1):
		slope_df[i] = df[cut*(i+1)] - df[cut*i] 
	#print slope_df
	return slope_df
	#slope_df.to_csv("slope_" + str(cut) + ".csv", sep=',', index=None)
	#draw_line(slope_df.ix[5], cut)
	#draw_line(df.ix[5])
	#draw_line(slope_df.ix[0])
def load_slope_df(cut):
	slope_df = pd.read_csv("slope_" + str(cut) + ".csv", sep=',')
	feature_array = np.empty([len(slope_df), len(slope_df.ix[0])])
	print slope_df.head()
	for i in range(len(slope_df)):
		feature_array[i] = np.array(slope_df.ix[i])
	print feature_array.shape
	return feature_array

def draw_mds(feature_array, labels):
	mds = manifold.MDS(2, max_iter=100, n_init=1)
	new_feature = mds.fit_transform(feature_array)
	print new_feature.shape
	import matplotlib.pyplot as plt
	plt.scatter(new_feature[:, 0], new_feature[:, 1], c = labels)
	plt.show()

def draw_orign_mds(power_df):
	#feature_array = np.empty([len(power_df), len(power_df.ix[0])])
	feature_array = power_df.as_matrix()
	print feature_array.shape
	mds = manifold.MDS(2, max_iter=100, n_init=1)
	new_feature = mds.fit_transform(feature_array)
	import matplotlib.pyplot as plt
	plt.scatter(new_feature[:, 0], new_feature[:, 1])
	plt.show()


def kmeans(power_df, cluster = 3):
	clf = KMeans(n_clusters=cluster)
	feature_array = power_df.as_matrix()
	min_max_scaler = preprocessing.MinMaxScaler()
	for i in range(len(feature_array)):
		feature_array[i] = min_max_scaler.fit_transform(feature_array[i])
	draw_line(feature_array)
	result = clf.fit(feature_array)
	#draw_legend(feature_array, clf.labels_)
	draw_mds(feature_array, clf.labels_)
	power_df = filter_NA()
	power_df['label'] = clf.labels_
	power_df.to_csv("power_day_48_kmeans_labels_"+str(cluster)+".csv", sep=',', index=None)
	return clf.labels_

def dbscan(power_df,cluster = 3):
	clf = DBSCAN(eps=1, min_samples=3)
	feature_array = power_df.as_matrix()
	min_max_scaler = preprocessing.MinMaxScaler()
	for i in range(len(feature_array)):
		feature_array[i] = min_max_scaler.fit_transform(feature_array[i])
	draw_line(feature_array)
	result = clf.fit(feature_array)

	draw_mds(feature_array, clf.labels_)
	power_df = filter_NA()
	power_df['label'] = clf.labels_
	num = np.max(clf.labels_)
	print "label num", num
	power_df.to_csv("power_day_48_dbscan_labels_"+str(num)+".csv", sep=',', index=None)
	for i in range(num):
		cur_df = power_df[power_df['label'] == i]
		#print cur_df.head()
		cur_df = cur_df.drop('label', 1)
		lines = cur_df.as_matrix()
		print len(lines)
		draw_line(lines)

	cur_df = power_df[power_df['label'] == -1]
	cur_df = cur_df.drop('label', 1)
	lines = cur_df.as_matrix()
	print len(lines)
	draw_line(lines)

	return clf.labels_


def filter_NA():
	power_df = pd.read_csv("./power_day_48.csv", header=None, na_values=0)
	power_df = power_df.dropna()
	print len(power_df)
	return power_df


def draw_lines(cluster=3):
	power_df = pd.read_csv("power_day_48_kmeans_labels_"+ str(cluster) + ".csv", sep=',')
	#print power_df.head()
	for i in range(cluster):
		cur_df = power_df[power_df['label'] == i]
		#print cur_df.head()
		cur_df = cur_df.drop('label', 1)
		lines = cur_df.as_matrix()
		print len(lines)
		draw_line(lines)
		#print len(cur_df)
		#print cur_df.head()
	return


if __name__ == "__main__":

	#修改编码
	reload(sys)
	sys.setdefaultencoding("utf-8")

	print "Processing..."

	#宏变量
	POINT_NUM = 48
	SAMPLE_NUM = 0
	power_df = filter_NA()
	lines = power_df.as_matrix()
	df = cal_slope(power_df, cut=1)
	dbscan(power_df)
	# kmeans(df, cluster=7)
	
	# draw_lines(cluster=7)

	
	

