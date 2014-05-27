import numpy as np
from scipy.cluster.vq import vq, kmeans2
from scipy.io import wavfile
from librosa import logamplitude
from librosa.feature import delta, melspectrogram
from librosa.filters import dct


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
    """Mel-frequency cepstral coefficients

    :usage:
        >>> # Generate mfccs from a time series
        >>> mfccs = librosa.feature.mfcc(y=y, sr=sr)

        >>> # Use a pre-computed log-power Mel spectrogram
        >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        >>> mfccs = librosa.feature.mfcc(S=librosa.logamplitude(S))

        >>> # Get more components
        >>> mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

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


def build_codebook(data, k):
    codebook, _ = kmeans2(data, k)
    return codebook


def build_codebooks_from_list_of_wav(wavs, ks):
    mfccs = []
    d_mfccs = []
    dd_mfccs = []
    for w in wavs:
        print("preprocessing {}".format(w))
        sr, data = wavfile.read(w)
        cur_mfccs = mfcc(data, sr=sr, **MFCC_PARAMS)
        mfccs.append(cur_mfccs.T)
        d_mfccs.append(delta(cur_mfccs).T)
        dd_mfccs.append(delta(cur_mfccs, order=2).T)
    print("Building codebooks...")
    return (build_codebook(np.vstack(mfccs), ks[0]),
            build_codebook(np.vstack(d_mfccs), ks[1]),
            build_codebook(np.vstack(dd_mfccs), ks[2]))


def coocurrences(quantized_data, n_quantized, lag):
    pair_idx = quantized_data[lag:] * n_quantized + quantized_data[:-lag]
    return np.bincount(pair_idx, minlength=n_quantized ** 2)


def compute_coocurrences(data, centroids, lags):
    quantized, _ = vq(data, centroids)
    coocs = [coocurrences(quantized, centroids.shape[0], l) for l in lags]
    return np.hstack(coocs)


def hac(data, sr, codebooks, lags=[5, 2]):
    mfccs = mfcc(y=data, sr=sr, **MFCC_PARAMS)
    d_mfccs = delta(mfccs)
    dd_mfccs = delta(mfccs, order=2)
    streams = [mfccs.T, d_mfccs.T, dd_mfccs.T]
    return np.hstack([compute_coocurrences(stream, codebook, lags)
                      for (stream, codebook) in zip(streams, codebooks)
                      ])


def wav2hac(wav_path, codebooks, lags=[5, 2]):
    sr, data = wavfile.read(wav_path)
    return hac(data, sr, codebooks, lags=lags)
