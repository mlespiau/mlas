import numpy

class Cluster():
    def __init__(self, name, gmm, initialSegment):
        self.gmm = gmm
        self.name = name
        self.segments = []
        self.add_segment(initialSegment)

    def trainGmm(self):
        # TODO: specify number of iterations in EM algorithm?
        # TODO: is this using EM?
        # TODO: relevant code: max_em_iters=em_iters in gmm training
        self.gmm.fit(self.getAllSegmentsData())

    def get_gmm(self):
        return self.gmm

    def reset_data(self):
        self.segments = []

    def add_segment(self, segment):
        self.segments.append(segment)

    def getAllSegmentsData(self):
        data = numpy.array(self.segments[0].getData())
        for i in range(1, len(self.segments)):
            numpy.concatenate((data, self.segments[i].getData()))
        return data

    def getSegments(self):
        return self.segments

    def bic(self):
        return self.gmm.bic(self.getAllSegmentsData())

    def bic2(self):
        return (-2 * self.gmm.score(self.getAllSegmentsData()).sum() - self.gmm._n_parameters() * numpy.log(self.getAllSegmentsData().shape[0]))

    def score(self):
        return self.gmm.score(self.getAllSegmentsData())

    def get_name(self):
        return self.name

class Segment():
    def __init__(self, start, end, data):
        self.start = start
        self.end = end
        self.most_likely_gmm_class = None
        print 'Creating segment [' + str(start) + ':' + str(end) + ']. Class: ' + str(self.most_likely_gmm_class)
        self.data = numpy.array(data)

    def getData(self):
        return self.data

    def set_most_likely_gmm_class(self, gmmClass):
        self.most_likely_gmm_class = gmmClass

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end
