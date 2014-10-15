import scipy
import numpy
import math
from scipy.fftpack import dct
from scipy.io import wavfile
# from scipy.signal import wiener
from sklearn.mixture import GMM
# TODO: averiguar como funciona esto
import scipy.stats.mstats as stats

(rate,signal) = wavfile.read("corpus/telam-51aniosrayuela_part1.wav")

# TODO: add wiener filtering to the signal!
import numpy
from features import sigproc
from scipy.fftpack import dct

def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between seccessive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate)
    pspec = sigproc.powspec(frames,nfft)
    energy = numpy.sum(pspec,1) # this stores the total energy in each frame

    fb = get_filterbanks(nfilt,nfft,samplerate)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    return feat,energy

def logfbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between seccessive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
    return numpy.log(feat)

def ssc(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
    """Compute Spectral Subband Centroid features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between seccessive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate)
    pspec = sigproc.powspec(frames,nfft)

    fb = get_filterbanks(nfilt,nfft,samplerate)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    R = numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(pspec,1)),(numpy.size(pspec,0),1))

    return numpy.dot(pspec*R,fb.T) / feat

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1+hz/700.0)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([nfilt,nfft/2+1])
    for j in xrange(0,nfilt):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra,L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22.
    """
    nframes,ncoeff = numpy.shape(cepstra)
    n = numpy.arange(ncoeff)
    lift = 1+ (L/2)*numpy.sin(numpy.pi*n/L)
    return lift*cepstra


class Cluster():
    def __init__(self, gmm, data):
        self.gmm = gmm
        self.data = data

    def train(self):
        # TODO: specify number of iterations in EM algorithm?
        # TODO: is this using EM?
        # TODO: relevant code: max_em_iters=em_iters in gmm training
        self.gmm.fit(self.data)

    def getGmm():
        return self.gmm

class Segment():
    def __init__(self, gmm, mostLikelyGmmClass, data):
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
    def __init__(self, X, N, gmmList):
        self.X = X
        self.N = N
        self.gmmList = gmmList
        self.numberOfClusters = len(self.gmmList)
        self.intervalSize = 200

    def execute(self):
        likelihoods = self.gmmList[0].score(self.X)
        for gmm in self.gmmList[1:]:
            likelihoods = numpy.column_stack((likelihoods, gmm.score(self.X)))
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
                segment = Segment(self.gmmList[mostLikelyGmmClass], mostLikelyGmmClass, currentSegmentData)
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
print(X.shape)
N = X.shape[0]
D = X.shape[1]

# divide the features into k clusters
rowsPerCluster = N/numberOfCluster
gaussianComponents = 5

initialTraining = []
dataSplits = numpy.vsplit(X, range(rowsPerCluster, N, rowsPerCluster))
gmmList = []

# train a GMM in each cluster
for data in dataSplits:
    print 'Training GMM'
    gmm = GMM(n_components=gaussianComponents)
    gmmList.append(gmm)
    initialTraining.append(Cluster(gmm, data))
    print 'Done!'

for cluster in initialTraining:
    cluster.train()

NUMBER_INITIAL_SEGMENTATION_LOOPS = 2
NUMBER_SEGMENTATION_LOOPS = 3

resegmenter = Resegmenter(X, N, gmmList)
for iterationNumber in range(0, NUMBER_INITIAL_SEGMENTATION_LOOPS):
    resegmenter.execute()

# agregar getter a resegmenter para los reSegmentedClusters
bestBicScore = 1.0
while(bestBicScore > 0 and len(gmmList) > 1):
    resegmenter = Resegmenter(X, N, gmmList)
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
            score = 0.0
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
            if (score > bestBicScore):
                bestMergedGmm = newGmm
                mergedTuple = (clusterOne, clusterTwo)
                mergedTupleIndices = (gmmOneClusterName, gmmTwoClusterName)
                bestBicScore = score
    if bestBicScore > 0.0:
        gmmList.__delitem__(mergedTupleIndices[0])
        gmmList.__delitem__(mergedTupleIndices[1])
        gmmList[mergedTupleIndices[0]] = bestMergedGmm
    print(gmmList)
