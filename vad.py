import scipy
import numpy
import math
from features import mfcc
from scipy.io import wavfile
from scipy.signal import wiener
from sklearn.mixture import GMM

# % This Speech Activity Detector (SAD) is completely unsupervised and is based
# % on the speaker diarization system developed by NTU, Singapore for the
# % NIST RT-09 evaluation

# % Code written by Parthe Pandit
# % Date: 06-04-2015

# (rate,signal) = wavfile.read("corpus/marian-campos-electroestaticos-mono.wav")
(rate,signal) = wavfile.read("corpus/telam-51aniosrayuela_part1.wav")

signal_wiener_filtered = wiener(signal, noise=None)
print signal[0]
print signal_wiener_filtered[0]

#features typically used 20 MFCC with energy. no delta's
features = mfcc(signal, samplerate=rate, numcep=36, winlen=0.03,winstep=0.02)

# energy 1st MF Cepstral Coefficient
energy = features[:,0]

# indexing after sorting in ascending order based on energy
# energy.sort()
energy_sort_a_index = numpy.argsort(energy)
energy_sort_d_index = energy_sort_a_index[::-1]

ten_percent_highest_energy_index = energy_sort_d_index[0:int(len(features)*.1)]
twenty_percent_lowest_energy_index = energy_sort_a_index[0:int(len(features)*.2)]

sp = features[ten_percent_highest_energy_index][:]
nsp = features[twenty_percent_lowest_energy_index][:]

print 'len sp: ' + str(len(sp))
print 'len nsp:' + str(len(nsp))
# % train initial models of speech and non-speech using functions from the
# % cluster (Purdue) MATLAB toolbox
Gsp = GMM(n_components=16)
Gsp.fit(sp)
Gnsp = GMM(n_components=4)
Gnsp.fit(nsp)

I_old = len(sp);
J_old = len(nsp);
percent_change = 100;
counter = 0;

while(percent_change > 1):
    counter = counter + 1;
    print 'counter: ' + str(counter)

    # compute likelihood that frame is speech
    LLsp = Gsp.score(features)
    # compute likelihood that frame is non-speech
    LLnsp = Gnsp.score(features)

    print len(features)

    # get data for next iteration of training both models
    sp = features[LLsp > LLnsp,:]
    nsp = features[LLnsp > LLsp,:]

    # re-train the models for speech and non-speech
    Gsp = GMM(n_components=32)
    Gsp.fit(sp)
    Gnsp = GMM(n_components=8)
    Gnsp.fit(nsp)

    I_new = len(sp);
    J_new = len(nsp);

    print 'I_old: ' + str(I_old)
    print 'J_old: ' + str(J_old)
    print 'I_new: ' + str(I_new)
    print 'J_new: ' + str(J_new)

    sub = abs(I_new - I_old) * 1.0
    print 'sub: ' + str(sub)
    div = sub/I_old * 1.0
    print 'div : ' + str(div)
    percent_change = div * 100
    print 'percent_change:' + str(percent_change)
    I_old = I_new
    J_old = J_new

likelihood = LLnsp > LLsp
print likelihood
if likelihood[0] == True:
    previousType = 'nsp'
else:
    previousType = 'sp'
fromIndex = 0
toIndex = 0


for i in range(1, len(likelihood)):
    if likelihood[i] == True:
        currentType = 'nsp'
    else:
        currentType = 'sp'
    # print currentType
    # print previousType
    if currentType == previousType or (toIndex - fromIndex < 35):
        toIndex = i
    else:
        print 'Length: ' + str(toIndex - fromIndex) + ' From: ' + str(fromIndex) + ' To: ' + str(toIndex) + ' Type: ' + currentType
        fromIndex = i
        toIndex = i
        previousType = currentType
print 'Length: ' + str(toIndex - fromIndex) + ' From: ' + str(fromIndex) + ' To: ' + str(toIndex) + ' Type: ' + currentType



# % In the end:
# % LLsp and LLnsp contain the likelihood for each frame being speech and
# % non-speech respectively

# %% Continuity constraint

# % J_start and J_stop indicate the start and end frame-index of each non-speech
# % segment. The divide by 100 is because frame_hop is 10ms = 1/100 s

# min_length = 35; % in frames.
# % Thus only indicate to me segments that are contiguously
# % classified as non-speech and are longer than 350ms

# j = 2;
# count = 1;
# remove = 0;
# J_start = [];
# J_stop = [];
# J = [];

# L = LLsp - LLnsp;

# while( j < length(L))
#     if (L(j) < 0 && L(j-1)>=0)
#         start = j;
#     elseif (L(j) < 0)
#         count = count + 1;
#     elseif (L(j) > 0)
#         if (count > min_length)

#             J_start = [J_start start];
#             J_stop = [J_stop start + count];

#         end
#         count = 1;
#     end
#     j = j + 1;
# end

# for k = 1:length(J_start)
#     J = [J ; (J_start(k):min(length(L),J_stop(k)))'];
# end

# I = (1:length(L))';
# I(J) = [];

# disp('Done Final Classification !!');
# disp([J_start ;J_stop ; J_stop - J_start]'/100)

