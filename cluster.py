import numpy
import scipy.stats.mstats as stats

class Cluster():
    def __init__(self, name, gmm, initialSegment):
        self.gmm = gmm
        self.name = name
        self.segments = []
        self.add_segment(initialSegment)

    def train_gmm(self):
        # TODO: specify number of iterations in EM algorithm?
        # TODO: is this using EM?
        # TODO: relevant code: max_em_iters=em_iters in gmm training
        self.gmm.fit(self.get_all_segments_data())

    def get_gmm(self):
        return self.gmm

    def reset_data(self):
        self.segments = []

    def add_segment(self, segment):
        self.segments.append(segment)

    def get_all_segments_data(self):
        data = numpy.array(self.segments[0].get_data())
        for i in range(1, len(self.segments)):
            numpy.concatenate((data, self.segments[i].get_data()))
        return data

    def get_segments(self):
        return self.segments

    def bic(self):
        return self.gmm.bic(self.get_all_segments_data())

    def bic2(self):
        return (-2 * self.gmm.score(self.get_all_segments_data()).sum() - self.gmm._n_parameters() * numpy.log(self.get_all_segments_data().shape[0]))

    def score(self):
        return self.gmm.score(self.get_all_segments_data())

    def get_name(self):
        return self.name

class Segment():
    def __init__(self, start, end, data, most_likely_gmm_class):
        self.start = start
        self.end = end
        if most_likely_gmm_class:
            self.most_likely_gmm_class = most_likely_gmm_class
        else:
            raise Exception("noGmmClassForSegment")
        print 'Creating segment [' + str(start) + ':' + str(end) + ']. Class: ' + str(self.most_likely_gmm_class)
        self.data = numpy.array(data)

    def get_data(self):
        return self.data

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

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
            segment = Segment(
                data_range[i],
                data_range[i+1],
                current_segment_data,
                self.cluster_list[most_likely_gmm_class].get_name()
            )
            self.cluster_list[most_likely_gmm_class].add_segment(segment)
        new_cluster_list = []
        for cluster in self.cluster_list:
            if len(cluster.get_segments()) > 0:
                cluster.train_gmm()
                new_cluster_list.append(cluster)
        return new_cluster_list
