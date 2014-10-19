import scipy
import numpy
import math
from features import mfcc
from scipy.io import wavfile
# from scipy.signal import wiener
from sklearn.mixture import GMM
# TODO: averiguar como funciona esto
import scipy.stats.mstats as stats

# TODO: add wiener filtering to the signal!
(rate,signal) = wavfile.read("corpus/telam-51aniosrayuela_part1.wav")

class Cluster():
    def __init__(self, gmm, data):
        self.gmm = gmm
        self.data = data

    def train(self):
        # TODO: specify number of iterations in EM algorithm?
        # TODO: is this using EM?
        # TODO: relevant code: max_em_iters=em_iters in gmm training
        self.gmm.fit(self.data)

    def getGmm(self):
        return self.gmm

class Segment():
    def __init__(self, start, end, gmm, mostLikelyGmmClass, data):
        self.start = start
        self.end = end
        self.gmm = gmm
        self.mostLikelyGmmClass = mostLikelyGmmClass
        self.data = numpy.array(data)

    def getGmm(self):
        return self.gmm

    def getData(self):
        return self.data

    def addData(self, data):
        self.data = numpy.concatenate((self.data, data))

    def trainGmm(self):
        self.gmm.fit(self.data)

    def bic(self):
        return self.gmm.bic(self.data)

class Resegmenter():
    def __init__(self, X, N, cluster_list):
        self.X = X
        self.N = N
        self.cluster_list = cluster_list
        self.numberOfClusters = len(self.cluster_list)
        self.intervalSize = 200

    def execute(self):
        likelihoods = self.cluster_list[0].getGmm().score(self.X)
        for cluster in self.cluster_list[1:]:
            likelihoods = numpy.column_stack((likelihoods, cluster.getGmm().score(self.X)))
        if self.numberOfClusters == 1:
            self.mostLikely = numpy.zeros(len(self.X))
        else:
            self.mostLikely = likelihoods.argmax(axis=1)
        # Across 250 frames of observations
        # Vote on wich cluster they should be associated with
        self.reSegmentedClusters = {}
        dataRange = range(0, self.N, self.intervalSize)
        if dataRange[-1] < self.N:
            dataRange.append(self.N)
        print(len(self.mostLikely))
        print(dataRange)
        for i, v in enumerate(dataRange[0:len(dataRange)-1]):
            currentSegmentIndexes = range(dataRange[i], dataRange[i+1])
            currentSegmentScores = numpy.array(self.mostLikely[currentSegmentIndexes])
            # print(currentSegmentData)
            # TODO: averiguar como funciona esto stats.mode
            mostLikelyGmmClass = int(stats.mode(currentSegmentScores)[0][0])
            # print(mostLikelyGmmClass)
            # print(self.X[currentSegmentIndexes,:])
            currentSegmentData = self.X[currentSegmentIndexes,:]
            if (mostLikelyGmmClass in self.reSegmentedClusters):
                segment = self.reSegmentedClusters[mostLikelyGmmClass]
                segment.addData(currentSegmentData)
            else:
                segment = Segment(dataRange[i], dataRange[i+1], self.cluster_list[mostLikelyGmmClass].getGmm(), mostLikelyGmmClass, currentSegmentData)
                self.reSegmentedClusters[mostLikelyGmmClass] = segment
        # for each gmm, append all the segments and retrain
        # hay que verificar si lo que esta haciendo es justar en cluster_data
        # los datos de cada tipo p de cluster
        # iter_bic_dict e iter_bic_list hay que ver para que se usan para
        # entender mejor que son
            # iter_bic_dict se usa solo para kl distance
            # iter_bic_list se usa y es una lista de tuplas, gmm -> datos de ese cluster
            # most_likely se devuelve en el resegmenter solo para devolverlo al final de cluter
        print(self.reSegmentedClusters)
        for clusterName in self.reSegmentedClusters:
            self.reSegmentedClusters[clusterName].trainGmm()

    def getReSegementedClusters(self):
        return self.reSegmentedClusters

numberOfCluster = 16

X = mfcc(signal, samplerate=rate)
N = X.shape[0]
D = X.shape[1]

# divide the features into k clusters
rowsPerCluster = N/numberOfCluster
gaussianComponents = 5

cluster_list = []
dataSplits = numpy.vsplit(X, range(rowsPerCluster, N, rowsPerCluster))

# train a GMM in each cluster
for data in dataSplits:
    print 'Training GMM'
    gmm = GMM(n_components=gaussianComponents)
    cluster_list.append(Cluster(gmm, data))
    print 'Done!'

for cluster in cluster_list:
    cluster.train()

NUMBER_INITIAL_SEGMENTATION_LOOPS = 2
NUMBER_SEGMENTATION_LOOPS = 3

resegmenter = Resegmenter(X, N, cluster_list)
for iterationNumber in range(0, NUMBER_INITIAL_SEGMENTATION_LOOPS):
    resegmenter.execute()

# agregar getter a resegmenter para los reSegmentedClusters
bestBicScore = 1.0
while(bestBicScore > 0 and len(cluster_list) > 1):
    resegmenter = Resegmenter(X, N, cluster_list)
    for iterationNumber in range(0, NUMBER_SEGMENTATION_LOOPS):
        resegmenter.execute()
    bestMergedGmm = None
    bestBicScore = 0.0
    mergedTuple = None
    mergedTupleIndices = None
    clusters = resegmenter.getReSegementedClusters()
    print('Clusters...')
    print(clusters)
    clustersNames = clusters.keys()
    for i, gmmOneClusterName in enumerate(clustersNames):
        for j in range(i + 1, len(clustersNames)):
            gmmTwoClusterName = clustersNames[j]
            print 'gmmOneClusterName: ' + str(gmmOneClusterName) + '. gmmTwoClusterName: ' + str(gmmTwoClusterName)
            newScore = 0.0
            clusterOne = clusters[gmmOneClusterName]
            clusterTwo = clusters[gmmTwoClusterName]
            newClusterData = numpy.concatenate((clusterOne.getData(), clusterTwo.getData()))
            oneNumberOfComponents = clusterOne.getGmm().get_params()['n_components']
            twoNumberOfComponents = clusterTwo.getGmm().get_params()['n_components']
            newNumberOfComponents = oneNumberOfComponents + twoNumberOfComponents
            rationOne = float(oneNumberOfComponents) / float(newNumberOfComponents)
            rationTwo = float(twoNumberOfComponents) / float(newNumberOfComponents)
            w = numpy.ascontiguousarray(numpy.append(rationOne * clusterOne.getGmm().weights_, rationTwo * clusterTwo.getGmm().weights_))
            m = numpy.ascontiguousarray(numpy.append(clusterOne.getGmm().means_, clusterTwo.getGmm().means_))
            c = numpy.ascontiguousarray(numpy.append(clusterOne.getGmm().covars_, clusterTwo.getGmm().covars_))
            newGmm = GMM(n_components=newNumberOfComponents)
            newGmm.weights_ = w
            newGmm.means_ = m
            newGmm.covars_ = c
            newGmm.fit(newClusterData)
            # Esto podria ser fruta
            newScore = newGmm.bic(newClusterData) - clusterOne.bic() + clusterTwo.bic()
            print(newScore)
            if (newScore > bestBicScore):
                bestMergedGmm = newGmm
                mergedTuple = (clusterOne, clusterTwo)
                mergedTupleIndices = (gmmOneClusterName, gmmTwoClusterName)
                bestBicScore = newScore
                bestNewClusterData = newClusterData
    if bestBicScore > 0.0:
        print(mergedTupleIndices)
        print(cluster_list)
        if mergedTupleIndices[0] < mergedTupleIndices[1]:
            cluster_list.__delitem__(mergedTupleIndices[1])
            cluster_list.__delitem__(mergedTupleIndices[0])
        else:
            cluster_list.__delitem__(mergedTupleIndices[0])
            cluster_list.__delitem__(mergedTupleIndices[1])
        cluster_list.append(Cluster(bestMergedGmm, bestNewClusterData))
