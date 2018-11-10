import zlib
from abc import ABCMeta, abstractmethod
from functools import partial

import cytoolz as tz
import cytoolz.curried as tzc
import gensim.utils
import numpy as np
import scipy.sparse as sps
from gensim.corpora import Dictionary, HashDictionary
from gensim.models import NormModel, TfidfModel, LogEntropyModel, LdaModel, LdaMulticore, HdpModel, FastText
from gensim.models.tfidfmodel import df2idf
from sklearn.base import BaseEstimator, TransformerMixin


__all__ = [
    'sbow_itercoords',
    'sbow2array',
    'GensimTransformer',
    'GensimBaseDictionary',
    'GensimDictionary',
    'GensimHashDictionary',
    'GensimNormalize',
    'GensimTfidf',
    'GensimLogEntropy',
    'GensimLda',
    'GensimHdp',
    'GensimFastText'
]


def unzip(x):
    return zip(*x)


def sbow_itercoords(sparse_docs):
    for i, doc in enumerate(sparse_docs):
        for j, value in doc:
            yield i, j, value


def sbow2array(sparse_docs, shape=None, sparse=False, binarize=False):
    """ Convert from Gensim "sparse bag-of-words" format to array
    
    Essential for transform methods in Gensim wrappers
    """
    coords = sbow_itercoords(sparse_docs)

    if binarize:
        coords = ((i, j, float(value > 0)) for i, j, value in coords)

    i, j, values = unzip(coords)

    if shape is None:
        shape = max(i)+1, max(j)+1
    else:
        nrow, ncol = shape
        if nrow is None or nrow < 0:
            nrow = max(i)+1
        if ncol is None or ncol < 0:
            ncol = max(j)+1

    if sparse:
        out = sps.coo_matrix((values, (i, j)), (nrow, ncol))
    else:
        out = np.zeros((nrow, ncol))
        out[i, j] = values

    return out


def array2sbow(array, zero_tol=1e-07):
    """ Convert from Gensim "sparse bag-of-words" format to array
    
    This isn't actually needed for wrapping Gensim
    """
    if sps.issparse(array):
        array = array.tocoo()
        coo_dta = zip(array.row, array.col, array.data)
        for _, grp in it.groupby(coo_dta, key=tzc.get(0)):
            yield tuple((j, value) for _, j, value in grp)
    else:
        for row in array:
            yield tuple((j, value) for j, value in enumerate(row) if abs(value) < zero_tol)


# GensimTfidf(wlocal=tfidf_local_bm25)
# https://en.wikipedia.org/wiki/Okapi_BM25
def tfidf_local_bm25(termfreq, doclen, avgdoclen, k1=1.2, b=0.75):
    return termfreq * (k1 + 1) / (termfreq + k1*(1-b + doclen/avgdoclen))


# GensimTfidf(wglobal=tfidf_global_bm25)
# https://en.wikipedia.org/wiki/Okapi_BM25
def tfidf_global_bm25(docfreq, totaldocs):
    return np.log(totaldocs - docfreq + 0.5) - np.log(docfreq + 0.5)


# GensimTfidf(wlocal=tfidf_global_bm25plus)
# https://en.wikipedia.org/wiki/Okapi_BM25
# See
#   "Lower-Bounding Term Frequency Normalization"
#   Lanhua Lv and ChengXiang Zhai
#   http://sifaka.cs.uiuc.edu/czhai/pub/cikm11-bm25.pdf
def tfidf_global_bm25plus(termfreq, doclen, avgdoclen, k1=1.2, b=0.75, delta=1.0):
    return termfreq * (k1 + 1) / (termfreq + k1*(1-b + doclen/avgdoclen)) + delta


class GensimTransformer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    @classmethod
    def from_gensim(cls, tfidfmodel):
        raise NotImplementedError()

    @abstractmethod
    def _new_model(self, X=None, y=None):
        pass

    def fit(self, X, y=None):
        self.model_ = self._new_model(X)
        return self

    @abstractmethod
    def transform(self, X):
        pass


class GensimBaseDictionary(GensimTransformer, metaclass=ABCMeta):
    @abstractmethod
    def _new_model(self, X=None, y=None):
        pass

    def partial_fit(self, X, y=None):
        if self.model_ is None:
            self.model_ = self._new_model(X)
        else:
            self.model_.add_documents(X)
        return self

    def fit(self, X, y=None):
        self.model_ = None
        return self.partial_fit(X, y)

    def transform(self, X, y=None):
        sbow = (self.model_.doc2bow(doc) for doc in X)
        if self.array_output:
            nrow = X.shape[0] if hasattr(X, 'shape') else len(X)
            ncol = len(self.model_)
            return sbow2array(sbow, shape=(nrow, ncol), sparse=self.sparse_output, binarize=self.binarize)
        else:
            return sbow


class GensimDictionary(GensimBaseDictionary):
    """ Minimal Scikit-learn wrapper for gensim.corpora.Dictionary
    
    no_below, no_above, keep_n, keep_tokens ignored if filter_extremes=False
    """
    def __init__(self, prune_at=2000000, array_output=True, sparse_output=True, binarize=False,
                 filter_extremes=False, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None):
        self.model_ = None
        self.prune_at = prune_at
        self.array_output = array_output
        self.sparse_output = sparse_output
        self.binarize = binarize
        self.filter_extremes = filter_extremes
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n
        self.keep_tokens = keep_tokens

    def _new_model(self, X=None, y=None):
        return Dictionary(X, prune_at=self.prune_at)

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.filter_extremes:
            self.model_.filter_extremes(self.no_below, self.no_above, self.keep_n, self.keep_tokens)
        return self

    def get_colnames(self):
        """ Get the column names that would be assigned to a DataFrame of the result """
        return list(map(tzc.get(1), sorted(self.model_.items(), key=tzc.get(0))))[:5]


class GensimHashDictionary(GensimBaseDictionary):
    """ Minimal Scikit-learn wrapper for gensim.corpora.HashDictionary """
    def __init__(self, id_range=32000, myhash=zlib.adler32, debug=True, array_output=True, sparse_output=True, binarize=False):
        self.model_ = None
        self.id_range = id_range
        self.myhash = myhash
        self.debug = debug
        self.array_output = array_output
        self.sparse_output = sparse_output
        self.binarize = binarize

    def _new_model(self, X=None, y=None):
        return HashDictionary(id_range=self.id_range, myhash=self.myhash, debug=self.debug)


class GensimNormalize(GensimTransformer):
    """ Minimal Scikit-learn wrapper for gensim.models.NormModel """
    def __init__(self, norm='l2'):
        self.model_ = None
        self.norm = norm

    def _new_model(self, X=None, y=None):
        return NormModel(norm=self.norm)

    def transform(self, X=None, y=None):
        # this one "knows what to do" with matrices, etc
        return self.model_[X]


class GensimTfidf(GensimTransformer):
    """ Minimal Scikit-learn wrapper for gensim.models.TfidfModel """
    def __init__(self, wlocal=gensim.utils.identity, wglobal=df2idf, normalize=True, smartirs=None, pivot=None, slope=0.65, array_output=True, sparse_output=True, id2word=None, vocab_size=None):
        self.model_ = None
        self.wlocal = wlocal
        self.wglobal = wglobal
        self.normalize = normalize
        self.smartirs = smartirs
        self.pivot = pivot
        self.slope = slope
        self.array_output = array_output
        self.sparse_output = sparse_output
        self.id2word = id2word
        self.vocab_size = vocab_size

    def _new_model(self, X=None, y=None):
        return TfidfModel(X, wlocal=self.wlocal, wglobal=self.wglobal, normalize=self.normalize,
                          smartirs=self.smartirs, pivot=self.pivot, slope=self.slope, id2word=self.id2word)

    def transform(self, X=None, y=None):
        if self.array_output:
            model = self.model_
            if model.id2word is not None:
                ncol = len(model.id2word)
            elif self.vocab_size is not None:
                ncol = self.vocab_size
            else:
                ncol = None
            return sbow2array(model[X], (None, ncol), sparse=self.sparse_output)
        else:
            return self.model_[X]


class GensimLogEntropy(GensimTransformer):
    """ Minimal Scikit-learn wrapper for gensim.models.LogEntropyModel
    
    A special case of TF-IDF with "log+1" local weight and entropy global weight
    """
    def __init__(self, normalize=True, array_output=True, sparse_output=False):
        self.model_ = None
        self.normalize = True
        self.array_output = array_output
        self.sparse_output = sparse_output

    def _new_model(self, X=None, y=None):
        return LogEntropyModel(X, normalize=self.normalize)

    def transform(self, X=None, y=None):
        if self.array_output:
            return sbow2array(self.model_[X], (None, len(self.model_.entr)+1), sparse=self.sparse_output)
        else:
            return self.model_[X]


class GensimLda(GensimTransformer):
    """ Minimal Scikit-learn wrapper for gensim.models.LdaModel """
    def __init__(self, num_topics=100, chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0,
                 eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01, ns_conf=None, minimum_phi_value=0.01,
                 callbacks=None, dtype=np.dtype('float32'), n_workers=1, random_state=None, array_output=True, sparse_output=True, id2word=None):
        self.model_ = None
        self.callbacks = callbacks
        self.dtype = dtype
        self.random_state = random_state
        self.n_workers = n_workers
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.update_every = update_every
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.eval_every = eval_every
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        self.minimum_probability = minimum_probability
        self.ns_conf = ns_conf
        self.minimum_phi_value = minimum_phi_value
        self.array_output = array_output
        self.sparse_output = sparse_output
        self.id2word = id2word

    def _new_model(self, X=None, y=None):
        if self.n_workers < 1:
            return LdaModel(X, num_topics=self.num_topics, chunksize=self.chunksize, passes=self.passes, batch=self.batch,
                            alpha=self.alpha, eta=self.eta, decay=self.decay, offset=self.offset, eval_every=self.eval_every, iterations=self.iterations, gamma_threshold=self.gamma_threshold,
                            minimum_probability=self.minimum_probability, minimum_phi_value=self.minimum_phi_value, per_word_topics=self.per_word_topics,
                            random_state=self.random_state, callbacks=self.self.callbacks, dtype=self.dtype)
        else:
            return LdaMulticore(X, num_topics=self.num_topics, chunksize=self.chunksize, passes=self.passes, batch=self.batch,
                            alpha=self.alpha, eta=self.eta, decay=self.decay, offset=self.offset, eval_every=self.eval_every, iterations=self.iterations, gamma_threshold=self.gamma_threshold,
                            minimum_probability=self.minimum_probability, minimum_phi_value=self.minimum_phi_value, per_word_topics=self.per_word_topics,
                            random_state=self.random_state, callbacks=self.self.callbacks, dtype=self.dtype)

    def partial_fit(self, X, y=None, chunks_as_numpy=False):
        if self.model_ is None:
            self.model_ = self._new_model(X, y)
        else:
            self.model_.update(X, chunks_as_numpy=chunks_as_numpy)
        return self

    def fit(self, X, y=None):
        self.model_ = self._new_model(X, y)
        return self

    def transform(self, X=None, y=None):
        if self.array_output:
            model = self.model_
            if model.id2word is not None:
                ncol = len(model.id2word)
            else:
                ncol = None
            return sbow2array(model[X], (None, ncol), sparse=self.sparse_output)
        else:
            return self.model_[X]


class GensimHdp(GensimTransformer):
    """ Minimal Scikit-learn wrapper for gensim.models.HdpModel """
    def __init__(self, max_chunks=None, max_time=None, chunksize=256, kappa=1.0, tau=64.0, K=15, T=150, alpha=1, gamma=1, eta=0.01, scale=1.0, var_converge=0.0001, outputdir=None, random_state=None, array_output=True, sparse_output=False):
        self.model_ = None
        self.max_chunks = max_chunks
        self.max_time = max_time
        self.chunksize = chunksize
        self.kappa = kappa
        self.tau = tau
        self.K = K
        self.T = T
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.scale = scale
        self.var_converge = var_converge
        self.outputdir_ = outputdir
        self.random_state = None
        self.array_output = array_output
        self.sparse_output = sparse_output

    def _new_model(self, X=None, y=None):
        return HdpModel(X, max_chunks=self.max_chunks, max_time=self.max_time, chunksize=self.chunksize, kappa=self.kappa, tau=self.tau, K=self.K, T=self.T,
                        alpha=self.alpha, gamma=self.gamma, eta=self.eta, scale=self.scale, var_converge=self.var_converge, outputdir=self.outputdir, random_state=self.random_state)

    def partial_fit(self, X, y=None, update=True, opt_o=True):
        if self.model_ is None:
            self.model_ = self._new_model(X)
        self.model_.update_chunk(X, update=update, opt_o=opt_o)
        return self

    def fit(self, X, y=None):
        self.model_ = self._new_model(X, y)
        return self

    def transform(self, X=None, y=None):
        if self.array_output:
            model = self.model_
            if model.id2word is not None:
                ncol = len(model.id2word)
            else:
                ncol = None
            return sbow2array(model[X], (None, ncol), sparse=self.sparse_output)
        else:
            return self.model_[X]


class GensimFastText(GensimTransformer):
    """ Minimal Scikit-learn wrapper for gensim.models.FastText """
    def __init__(self, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, word_ngrams=1, sample=0.001, seed=1,
                 workers=3, min_alpha=0.0001, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1,
                 bucket=2000000, trim_rule=None, batch_words=10000, callbacks=(), array_output=True):
        self.model_ = None
        self.n_workers_ = n_workers
        self.seed_ = seed
        self.callbacks_ = callbacks
        self.sg = sg
        self.hs = hs
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.word_ngrams = word_ngrams
        self.sample = sample
        self.min_alpha = min_alpha
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.min_n = min_n
        self.max_n = max_n
        self.sorted_vocab = sorted_vocab
        self.bucket = bucket
        self.trim_rule = trim_rule
        self.batch_words = batch_words
        self.array_output=array_output

    def _new_model(self, X=None, y=None):
        return FastText(X, sg=self.sg, hs=self.hs, size=self.size, alpha=self.alpha, window=self.window, min_count=self.min_count, max_vocab_size=self.max_vocab_size, word_ngrams=self.word_ngrams, sample=self.sample,
                        min_alpha=self.self.self.min_alpha, negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word, min_n=self.min_n,
                        max_n=self.max_n, sorted_vocab=self.sorted_vocab, bucket=self.self.self.bucket, trim_rule=self.trim_rule, batch_words=self.batch_words,
                        callbacks=self.callbacks_, seed=self.seed_, workers=self.n_workers_)

    def fit(self, X, y=None):
        self.model_ = self._new_model(X, y)
        return self

    def transform(self, X=None, y=None):
        if self.array_output:
            model = self.model_
            if model.vocabulary is not None:
                ncol = len(model.vocabulary)
            else:
                ncol = None
            return sbow2array(model.wv[X], (None, ncol), sparse=self.sparse_output)
        else:
            return self.model_.wv[X]
