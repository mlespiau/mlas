import scipy
import numpy
import math
from features import mfcc
from scipy.io import wavfile
# from scipy.signal import wiener
from sklearn.mixture import GMM
import scipy.stats.mstats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from cluster import Cluster, Segment

# TODO: add wiener filtering to the signal!
(rate,signal) = wavfile.read("corpus/telam-51aniosrayuela_part1.wav")


class Resegmenter():
    def __init__(self, X, N, cluster_list):
        self.X = X
        self.N = N
        self.cluster_list = cluster_list
        self.number_of_clusters = len(self.cluster_list)
        self.intervalSize = 150

    def execute(self):
        likelihoods = self.cluster_list[0].getGmm().score(self.X)
        self.cluster_list[0].resetData()
        for cluster in self.cluster_list[1:]:
            likelihoods = numpy.column_stack((likelihoods, cluster.getGmm().score(self.X)))
            cluster.resetData()
        if self.number_of_clusters == 1:
            self.mostLikely = numpy.zeros(len(self.X))
        else:
            self.mostLikely = likelihoods.argmax(axis=1)
        # Across 250 frames of observations
        # Vote on wich cluster they should be associated with
        dataRange = range(0, self.N, self.intervalSize)
        if dataRange[-1] < self.N:
            dataRange.append(self.N)
        for i, v in enumerate(dataRange[0:len(dataRange)-1]):
            currentSegmentIndexes = range(dataRange[i], dataRange[i+1])
            currentSegmentScores = numpy.array(self.mostLikely[currentSegmentIndexes])
            # print(currentSegmentData)
            mostLikelyGmmClass = int(stats.mode(currentSegmentScores)[0][0])
            print(mostLikelyGmmClass)
            # print(self.X[currentSegmentIndexes,:])
            currentSegmentData = self.X[currentSegmentIndexes,:]
            segment = Segment(dataRange[i], dataRange[i+1], currentSegmentData)
            segment.setMostLikelyGmmClass(self.cluster_list[mostLikelyGmmClass].getName())
            self.cluster_list[mostLikelyGmmClass].addSegment(segment)
        new_cluster_list = []
        for cluster in self.cluster_list:
            if len(cluster.getSegments()) > 0:
                cluster.trainGmm()
                new_cluster_list.append(cluster)
        return new_cluster_list

def print_cluster_list(clusters):
    bounds = []
    for cluster in clusters:
        print 'Cluster: ' + str(cluster.getName())
        for segment in cluster.getSegments():
            # bounds.append(segment.getStart())
            print 'Segment: start: ' + str(segment.getStart()) + '. end: ' + str(segment.getEnd())

class ClusterNames():
    def __init__(self):
        self.index = 65

    def getNextName(self):
        name = chr(self.index)
        self.index = self.index + 1
        return name

number_of_clusters = 8

# estoy usando
X = mfcc(signal, samplerate=rate)
N = X.shape[0]
D = X.shape[1]
# print(N)
# pca = PCA(n_components=3)
# pca.fit(X)
# pcaData = pca.transform(X)
# print(pcaData[1:10,0])
# print(pcaData[1:10,1])
# # plt.plot(pcaData[:,0], pcaData[:,1], pcaData[:,2],, 'o')
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pcaData[:,0], pcaData[:,1], pcaData[:,2])
# plt.show()

# divide the features into k clusters
rows_per_cluster = N/number_of_clusters
gaussian_components = 5
clusterNames = ClusterNames()
cluster_list = []
dataSplits = numpy.vsplit(X, range(rows_per_cluster, N, rows_per_cluster))
j = 0
# train a GMM in each cluster
for data in dataSplits:
    clusterName = clusterNames.getNextName()
    print 'Training GMM ' + clusterName
    gmm = GMM(n_components=gaussian_components, covariance_type='full')
    cluster_list.append(Cluster(clusterName, gmm, Segment(j, j + rows_per_cluster, data)))
    j = j + rows_per_cluster
    print 'Done!'

for cluster in cluster_list:
    cluster.trainGmm()

NUMBER_INITIAL_SEGMENTATION_LOOPS = 2
NUMBER_SEGMENTATION_LOOPS = 3

for iterationNumber in range(0, NUMBER_INITIAL_SEGMENTATION_LOOPS):
    resegmenter = Resegmenter(X, N, cluster_list)
    cluster_list = resegmenter.execute()

print_cluster_list(cluster_list)

# agregar getter a resegmenter para los reSegmentedClusters
bestBicScore = 1.0
while(bestBicScore > 0 and len(cluster_list) > 1):
    resegmenter = Resegmenter(X, N, cluster_list)
    for iterationNumber in range(0, NUMBER_SEGMENTATION_LOOPS):
        cluster_list = resegmenter.execute()
    bestMergedGmm = None
    bestBicScore = 0.0
    mergedTuple = None
    mergedTupleIndices = None
    print('Clusters...')
    print_cluster_list(cluster_list)
    for i in range(0, len(cluster_list) - 1):
        for j in range(i + 1, len(cluster_list)):
            clusterOne = cluster_list[i]
            clusterTwo = cluster_list[j]
            print 'gmmOneClusterName: ' + str(clusterOne.getName()) + '. gmmTwoClusterName: ' + str(clusterTwo.getName())
            newScore = 0.0
            newClusterData = numpy.concatenate((clusterOne.getAllSegmentsData(), clusterTwo.getAllSegmentsData()))
            oneNumberOfComponents = clusterOne.getGmm().get_params()['n_components']
            twoNumberOfComponents = clusterTwo.getGmm().get_params()['n_components']
            newNumberOfComponents = oneNumberOfComponents + twoNumberOfComponents
#             rationOne = float(oneNumberOfComponents) / float(newNumberOfComponents)
#             rationTwo = float(twoNumberOfComponents) / float(newNumberOfComponents)
#             w = numpy.ascontiguousarray(numpy.append(rationOne * clusterOne.getGmm().weights_, rationTwo * clusterTwo.getGmm().weights_))
#             m = numpy.ascontiguousarray(numpy.append(clusterOne.getGmm().means_, clusterTwo.getGmm().means_))
#             c = numpy.ascontiguousarray(numpy.append(clusterOne.getGmm().covars_, clusterTwo.getGmm().covars_))
            newGmm = GMM(n_components=newNumberOfComponents, covariance_type='full')
#             newGmm.weights_ = w
#             newGmm.means_ = m
#             newGmm.covars_ = c
            newGmm.fit(newClusterData)
#             # This is an approximation for X Aguera. thesis p78
#             # lambda is not needed as #Mij = #Mi + #Mj
#             # FIXME: the algorithm is not converging and only stops when there is only one cluster :(
#             newGmmBic = -2 * newGmm.score(newClusterData).sum() - newGmm._n_parameters() * numpy.log(newClusterData.shape[0])
            newGmmBic = newGmm.bic(newClusterData)
            newScore = newGmmBic - clusterOne.bic() - clusterTwo.bic()
            print('newGmmScore: ' + str(newGmm.bic(newClusterData)))
            print('clusterOneScore: ' + str(clusterOne.bic()))
            print('clusterTwoScore: ' + str(clusterTwo.bic()))
            print('final: ' + str(newScore))
            if (newScore > bestBicScore):
                bestMergedGmm = newGmm
                mergedTuple = (clusterOne, clusterTwo)
                mergedTupleIndices = (i, j)
                bestBicScore = newScore
                bestNewSegments = clusterOne.getSegments() + clusterTwo.getSegments()
    if bestBicScore > 0.0:
        print(mergedTupleIndices)
        new_cluster = Cluster(clusterNames.getNextName(), bestMergedGmm, bestNewSegments[0])
        for i in range(1, len(bestNewSegments)):
            new_cluster.addSegment(bestNewSegments[i])
        if mergedTupleIndices[0] < mergedTupleIndices[1]:
            cluster_list.__delitem__(mergedTupleIndices[1])
            cluster_list.__delitem__(mergedTupleIndices[0])
        else:
            cluster_list.__delitem__(mergedTupleIndices[0])
            cluster_list.__delitem__(mergedTupleIndices[1])
        cluster_list.append(new_cluster)
        print_cluster_list(cluster_list)
