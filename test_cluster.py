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
        self.interval_size = 150

    def execute(self):
        likelihoods = self.cluster_list[0].get_gmm().score(self.X)
        self.cluster_list[0].reset_data()
        for cluster in self.cluster_list[1:]:
            likelihoods = numpy.column_stack((likelihoods, cluster.get_gmm().score(self.X)))
            cluster.reset_data()
        if self.number_of_clusters == 1:
            self.most_likely = numpy.zeros(len(self.X))
        else:
            self.most_likely = likelihoods.argmax(axis=1)
        # Across 250 frames of observations
        # Vote on wich cluster they should be associated with
        data_range = range(0, self.N, self.interval_size)
        if data_range[-1] < self.N:
            data_range.append(self.N)
        for i, v in enumerate(data_range[0:len(data_range)-1]):
            current_segment_indexes = range(data_range[i], data_range[i+1])
            current_segment_scores = numpy.array(self.most_likely[current_segment_indexes])
            # print(current_segment_data)
            most_likely_gmm_class = int(stats.mode(current_segment_scores)[0][0])
            print(most_likely_gmm_class)
            # print(self.X[current_segment_indexes,:])
            current_segment_data = self.X[current_segment_indexes,:]
            segment = Segment(data_range[i], data_range[i+1], current_segment_data)
            segment.set_most_likely_gmm_class(self.cluster_list[most_likely_gmm_class].get_name())
            self.cluster_list[most_likely_gmm_class].add_segment(segment)
        new_cluster_list = []
        for cluster in self.cluster_list:
            if len(cluster.get_segments()) > 0:
                cluster.train_gmm()
                new_cluster_list.append(cluster)
        return new_cluster_list

def print_cluster_list(clusters):
    bounds = []
    for cluster in clusters:
        print 'Cluster: ' + str(cluster.getName())
        for segment in cluster.get_segments():
            # bounds.append(segment.getStart())
            print 'Segment: start: ' + str(segment.getStart()) + '. end: ' + str(segment.getEnd())

class ClusterNames():
    def __init__(self):
        self.index = 65

    def getNextName(self):
        name = chr(self.index)
        self.index = self.index + 1
        return name

def vad(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True):
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

number_of_clusters = 8

# estoy usando
X = mfcc(signal, samplerate=rate)
# voice_activity_frames = vad(signal, samplerate=rate)

# frameVars = np.var(frames, 1)
# reducedFrames = frames[np.where(frameVars > signal)]
# return reducedFrames

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
    cluster.train_gmm()

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
            print 'gmmOneClusterName: ' + str(clusterOne.get_name()) + '. gmmTwoClusterName: ' + str(clusterTwo.get_name())
            newScore = 0.0
            newClusterData = numpy.concatenate((clusterOne.getAllSegmentsData(), clusterTwo.getAllSegmentsData()))
            oneNumberOfComponents = clusterOne.get_gmm().get_params()['n_components']
            twoNumberOfComponents = clusterTwo.get_gmm().get_params()['n_components']
            newNumberOfComponents = oneNumberOfComponents + twoNumberOfComponents
#             rationOne = float(oneNumberOfComponents) / float(newNumberOfComponents)
#             rationTwo = float(twoNumberOfComponents) / float(newNumberOfComponents)
#             w = numpy.ascontiguousarray(numpy.append(rationOne * clusterOne.get_gmm().weights_, rationTwo * clusterTwo.get_gmm().weights_))
#             m = numpy.ascontiguousarray(numpy.append(clusterOne.get_gmm().means_, clusterTwo.get_gmm().means_))
#             c = numpy.ascontiguousarray(numpy.append(clusterOne.get_gmm().covars_, clusterTwo.get_gmm().covars_))
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
                bestNewSegments = clusterOne.get_segments() + clusterTwo.get_segments()
    if bestBicScore > 0.0:
        print(mergedTupleIndices)
        new_cluster = Cluster(clusterNames.getNextName(), bestMergedGmm, bestNewSegments[0])
        for i in range(1, len(bestNewSegments)):
            new_cluster.add_segment(bestNewSegments[i])
        if mergedTupleIndices[0] < mergedTupleIndices[1]:
            cluster_list.__delitem__(mergedTupleIndices[1])
            cluster_list.__delitem__(mergedTupleIndices[0])
        else:
            cluster_list.__delitem__(mergedTupleIndices[0])
            cluster_list.__delitem__(mergedTupleIndices[1])
        cluster_list.append(new_cluster)
        print_cluster_list(cluster_list)
