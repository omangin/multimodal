import numpy as np
from scipy.cluster.vq import vq, kmeans2
from scipy.io import wavfile
from librosa import logamplitude
from librosa.feature import delta, melspectrogram
from librosa.filters import dct


"""Implementation of the histograms of acoustic co-occurrences (HAC)
from [VanHamme2008] and [Driesen2012].

.. [VanHamme2008] Van Hamme, Hugo "HAC-models: a Novel Approach to Continuous
                  Speech Recognition", Interspeech ISCA (2008)
.. [Driesen2012] Driesen, Joris "Discovering words in speech using matrix
                 factorization", PhD thesis (2012)
               
"""


# Default parmeters for MFCC used in Louis Matlab code
# fs       = 16000; : sample frequency in Hz
# ncep     = 13;    : number of cepstra
# win      = 320;   : window lenght in samples
# shf      = 160;   : window's shift in samples (default win/2)
# nf       = 30;    : number of filters in the filterbank
# fb_step  = 100;   : the filterbank's step [bandwidth] (use 100)
# nlf      = 0;     : number of low filter to throw away (default 0)
# deriv    = 0;     : compute the deltas (use 1) and deltas-deltas (use 2)
#                     (default 0)

MFCC_PARAMS = {
    'n_mfcc': 13,   # Librosa default is 20
    'n_fft': 320,   # Librosa default is 2048
    'hop_length': 160,  # Librosa default is 512
}
# N_MELS = 30  # Librosa default is 128


def mfcc(data, sr=22050, n_mfcc=20, **kwargs):
    """Mel-frequency cepstral coefficients (original function from librosa,
    modified to accept additional parameters).

    :parameters:
      - data  : np.ndarray or None
          audio time series
      - sr    : int > 0
          sampling rate of y
      - n_mfcc: int
          number of MFCCs to return

    .. note::
        additional keyword arguments are passed to the melspectrogram function.

    :returns:
      - M     : np.ndarray, shape=(n_mfcc, S.shape[1])
          MFCC sequence

    """
    S = logamplitude(melspectrogram(y=data, sr=sr, **kwargs))
    return np.dot(dct(n_mfcc, S.shape[0]), S)


def iterative_kmeans(data, k, alpha=0.01):
    """Iterative kmeans: split each cluster in two along principal direction
    and run kmeans, until k is reached. (See [Driesen2012], Alg. 1.1, p53)

    params:
        - k: int
            number of clusters
        - alpha: double
            distance of split from original centroid
    """
    centroids = data.mean(axis=0)[np.newaxis, :]
    while centroids.shape[0] < k:
        quantized, _ = vq(data, centroids)
        counts = np.bincount(quantized, minlength=centroids.shape[0])
        has_max_points = counts.argmax()
        # Get svd for points associated
        associated = (quantized == has_max_points).nonzero()[0]
        u, s, v = np.linalg.svd(data[associated, :])
        p = v[0, :] * np.sqrt(s[0])
        # Split centroid
        p1 = centroids[has_max_points] + alpha * p
        p2 = centroids[has_max_points] - alpha * p
        centroids = np.vstack([centroids[:has_max_points, :],
                               centroids[has_max_points + 1:, :],
                               p1,
                               p2,
                               ])
        # Run kmeans
        centroids, _ = kmeans2(data, centroids, minit='matrix')
    return centroids


def build_codebook(data, k, mode='raw'):
    if mode == 'raw':
        codebook, _ = kmeans2(data, k)
    elif mode == 'iterative':
        codebook = iterative_kmeans(data, k)
    else:
        raise(ValueError('Invalid mode, should be raw or iterative'))
    return codebook


def build_codebooks_from_list_of_wav(wavs, ks, mode='raw'):
    """Generates three codebooks of low level units from a list of wav files.

    The three codebooks corresponds to a quantization of MFCC vectors
    from the sound files as well as their first and second order time
    derivatives.

    :parameters:
        - ks: triple of int
            Number of elements in each code book.

    :returns:
        triple of codebooks as (k, d) arrays
    """
    mfccs = []
    for w in wavs:
        print("preprocessing {}".format(w))
        sr, data = wavfile.read(w)
        cur_mfccs = mfcc(data, sr=sr, **MFCC_PARAMS)
        mfccs.append(cur_mfccs)
        #mfccs.append(cur_mfccs.T)
        #d_mfccs.append(delta(cur_mfccs).T)
        #dd_mfccs.append(delta(cur_mfccs, order=2).T)
    print("Building codebooks:")
    print("- MFCC...")
    cdb_mfcc = build_codebook(np.vstack([m.T for m in mfccs]), ks[0], mode=mode)
                              ks[0], mode=mode)
    print("- Delta MFCC...")
    cdb_dmfcc = build_codebook(np.vstack([delta(m).T for m in mfccs]), ks[1], mode=mode)
                               ks[1], mode=mode)
    print("- Delta Delta MFCC...")
    cdb_ddmfcc = build_codebook(np.vstack([delta(m, order=2).T for m in mfccs]), ks[2], mode=mode)
            np.vstack([delta(m, order=2).T for m in mfccs]),
            ks[2], mode=mode)
    return (cdb_mfcc, cdb_dmfcc, cdb_ddmfcc)


def coocurrences(quantized_data, n_quantized, lag):
    """Computes coocurrences counts from a quantized time serie.

    A coocurrence of values a and b is the event of the value being observed
    observed at time t and the value b at time t + lag. It is represented as
    the pair (a, b) and coded as an int.

    :parameters:
        - quantized_data: one dimensional array
        - n_quantized: int
            size of the codebook, should be greater than max(quantized_data)
        - lag: int

    :returns:
        the vector of count that is of size (n_quantized^2,)
    """
    pair_idx = quantized_data[lag:] * n_quantized + quantized_data[:-lag]
    return np.bincount(pair_idx, minlength=n_quantized ** 2)


def compute_coocurrences(data, centroids, lags):
    quantized, _ = vq(data, centroids)
    coocs = [coocurrences(quantized, centroids.shape[0], l) for l in lags]
    return np.hstack(coocs)


def hac(data, sr, codebooks, lags=[5, 2]):
    """Histogram of acoustic coocurrence (see [VanHamme2008]).

    A vector of counts is returned instead of an actual histogram.

    :parameters:
        - data: time serie
        - sr: sample rate
        - codebooks: triple of codebooks
        - lags: a list of lags to use (the corresponding histograms are
            concatenated).
    """
    mfccs = mfcc(data, sr=sr, **MFCC_PARAMS)
    d_mfccs = delta(mfccs)
    dd_mfccs = delta(mfccs, order=2)
    streams = [mfccs.T, d_mfccs.T, dd_mfccs.T]
    return np.hstack([compute_coocurrences(stream, codebook, lags)
                      for (stream, codebook) in zip(streams, codebooks)
                      ])


def wav2hac(wav_path, codebooks, lags=[5, 2]):
    sr, data = wavfile.read(wav_path)
    return hac(data, sr, codebooks, lags=lags)
