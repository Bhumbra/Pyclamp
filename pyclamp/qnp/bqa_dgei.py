import numpy as np
import probayes as pb
import scipy as sp
import scipy.stats
try:
  import scipy.misc
  comb = scipy.misc.comb
except AttributeError:
  import scipy.special
  comb = scipy.special.comb
import tqdm
import multiprocessing
from joblib import Parallel, delayed

NEARLY_ZERO = 1e-300
MINIMUM_GAMMA = 1.
CONCURRENCY = multiprocessing.cpu_count()

#-------------------------------------------------------------------------------
def gampdf(X, g, l, _gm1=None, _gllplgg=None, _con_axis=0):
  isscalar = not isinstance(X, (np.ndarray, list))
  if isscalar:
    if X <= 0.:
      zs = np.maximum(g.shape, l.shape)
      return np.zeros(zs, dtype=float)
  if _gm1 is None: _gm1 = g-1.
  if _gllplgg is None: _gllplgg = g*np.log(l) + sp.special.gammaln(g)
  if isscalar:
    return np.exp(_gm1*np.log(X) - X/l - _gllplgg)

  # Single job call
  if isinstance(X, np.ndarray):
    neg = np.ravel(X <= 0.)
    Y = np.maximum(X, NEARLY_ZERO)
    g = np.maximum(g, MINIMUM_GAMMA)
    Y = np.exp(_gm1*np.log(Y) - Y/l - _gllplgg)
    Y[neg] = 0.
    return Y

  # Multi job call
  Y = Parallel(n_jobs=CONCURRENCY, prefer="threads")(
               delayed(gampdf)(x, g, l, _gm1, _gllplgg, _con_axis) 
               for x in X)
  return np.concatenate(Y, axis=_con_axis)

#-------------------------------------------------------------------------------
def sumprod(In0, In1, axis, keepdims=False, _expand=False):
  if not isinstance(In0, list) or not isinstance(In1, list):
    S = np.sum(In0*In1, axis=axis, keepdims=keepdims)
    if not _expand:
      return S
    return np.expand_dims(S, 0)

  S = Parallel(n_jobs=CONCURRENCY, prefer="threads")(
               delayed(sumprod)(in0, in1, axis=axis-1, 
                                keepdims=keepdims, _expand=True) 
               for in0, in1 in zip(In0, In1))
  return np.concatenate(S, axis=0)

#-------------------------------------------------------------------------------
def binomial(n, p, axis):
  def _binomial(i, _n, _lbc, _logp, _log1mp):
    return np.exp(lbc[i] + i*_logp + (_n-i)*_log1mp)
  I = list(range(n+1))
  logp = np.log(p)
  log1mp = np.log(1-p)
  lbc = np.log(comb(n, np.arange(n+1), exact=False, repetition=False))
  B = Parallel(n_jobs=CONCURRENCY, prefer="threads")(
               delayed(_binomial)(i, n, lbc, logp ,log1mp) for i in I)
  return np.concatenate(B, axis=axis)

#-------------------------------------------------------------------------------
def binopmf(i, n, p, _logp=None, _log1mp=None, _con_axis=1):
  lbc = np.log(comb(int(n), i, exact=False, repetition=False))
  return np.exp(lbc + np.array(i, dtype=float) * np.log(p) \
                    + np.array(n - i, dtype=float) * np.log(1-p))

#-------------------------------------------------------------------------------
def lqlf(x, i, n, q, g, eps, plims=[0.04, 0.96], pgb=None):

  # Evaluate latent parameters
  I = np.ravel(i)
  u = np.unique(I).tolist()
  means = np.array([np.mean(x[idx==I]) for idx in u])
  stdev = np.array([np.std(x[idx==I]) for idx in u])
  count = np.array([np.sum(idx == I)  for idx in u], dtype=float)
  sterr = stdev / np.sqrt(count)
  se = sterr[i]
  m = means[i]
  l = q / g
  nq = q * np.array(n, dtype=float)
  p = np.minimum(np.maximum(m / nq, plims[0]), plims[1])
  d = int(np.nonzero(np.array(n.shape)>1)[0])
  N = np.ravel(n).tolist()
  X = np.ravel(x).tolist()

  # Prefetch binomial coefficients and gammas, and initialise probs
  coefs = [None] * len(N)
  g_pdf = [None] * len(N)
  probs = [None] * len(N)
  slice0 = [slice(None) for _ in range(n.ndim)]
  slice0[d] = slice(0, 1)
  slicep = [slice(None) for _ in range(n.ndim)]
  if pgb is not None: pgb.init("Evaluating {} posteriors".format(len(N)), len(N)*2)
  for k, num in tqdm.tqdm(enumerate(N), "Prefetching {} convolutions".format(len(N)),
                          disable=pgb is not None):
    if pgb is not None: pgb.set(k, True)
  #for k, num in enumerate(N):    
    slicep[d] = slice(k, k+1) 
    pslice = p[tuple(slicep)]
    reshape = np.ones(n.ndim, dtype=int)
    reshape[d] = num+1
    counts = np.arange(num+1, dtype=int).reshape(reshape)
    #coefs[k] = scipy.stats.binom.pmf(counts, num, pslice)
    #coefs[k] = binopmf(np.ravel(counts).tolist(), num, pslice)
    coefs[k] = binopmf(counts, num, pslice)
    #g_pdf[k] = scipy.stats.gamma.pdf(x, g*counts, l) # doesn't handle <0 correctly
    g_pdf[k]= gampdf(X, g * counts, l)
    probs[k] = coefs[k][tuple(slice0)] * \
               scipy.stats.norm.pdf(x, loc=0., scale=eps)
  
  # Add binomial-weighted gamma pdfs
  slices = [slice(None) for _ in range(n.ndim)]
  slices[d] = slice(1, None)
  slices = tuple(slices)
  for k, num in tqdm.tqdm(enumerate(N), "Evaluating {} likelihoods".format(len(N)),
                          disable=pgb is not None):
    if pgb is not None: pgb.set(k+len(N), True)
    gslice = g_pdf[k][slices]
    cslice = coefs[k][slices]
    #dot_prod = np.sum(cslice * gslice, axis=d, keepdims=True)
    dot_prod = sumprod(list(cslice), list(gslice), axis=d, keepdims=True)
    probs[k] = probs[k] + dot_prod

  # Multiply log-likelihood array with scaled z-PDF
  logp_shape = list(probs[0].shape)
  logp_shape[d] = len(N)
  logp = np.empty(logp_shape, dtype=float)
  slices = [slice(None) for _ in range(len(logp_shape))]
  for k, num in enumerate(N):
    slices[d] = slice(k, k+1)
    logp[tuple(slices)] = np.log(probs[k] + NEARLY_ZERO) + \
                          scipy.stats.norm.logpdf(0., loc=0., scale=se)
  #logp -= logp.mean() # Offset to prevent underflow errors
  return logp

#-------------------------------------------------------------------------------
class BQA:
  plims = None
  nlims = None
  vlims = None

  num = None
  means = None
  polarity = None
  xi = None
  joint = None

#-------------------------------------------------------------------------------
  def __init__(self, nlims=[1, 16],  plims=[0.04, 0.96], vlims=[0.05, 1.0]):
    self.set_lims(nlims, plims, vlims)
    self.set_joint()

#-------------------------------------------------------------------------------
  def set_lims(self, nlims=[1, 16],  plims=[0.04, 0.96], vlims=[0.05, 1.0]):
    self.nlims = nlims
    self.plims = plims
    self.vlims = vlims
    self.glims = [1. / self.vlims[1]**2, 1. / self.vlims[0]**2]

#-------------------------------------------------------------------------------
  def set_data(self, data, eps=None):
    self.eps = eps
    self.data = data if isinstance(data, list) else [data]
    self.means = np.array([np.mean(subdata) for subdata in self.data])
    self.num = len(self.means)
    self.polarity = 1. if np.mean(np.hstack(data)) >= 0. else -1.
    if self.eps is None:
      tails = [subdata[subdata >= 0.] for subdata in self.data] if self.polarity > 0.\
              else [subdata[subdata >= 0.] for subdata in self.data]
      tails = np.hstack(tails)
      if not len(tails):
        self.eps = 1.
      else:
        self.eps = 1.253 * np.mean(tails)
    means = self.polarity * self.means
    plims = self.plims
    nqlims = [np.min(means), np.max(means)]
    if np.max(means) / np.min(means) < np.max(plims) / np.min(plims):
      nqlo = np.max(means) / np.max(plims)
      nqhi = np.min(means) / np.min(plims)
      nqlims = [np.minimum(nqlo, nqhi), np.maximum(nqlo, nqhi)]
    self.qlims = [np.min(nqlims)/np.max(self.nlims), np.max(nqlims)/np.min(self.nlims)]
    
    self.x = pb.RV('x', (-np.inf, np.inf))
    self.i = pb.RV('i', [0, self.num])
    self.stats = pb.RF(self.x, self.i)

    self.xi = [[], []]
    for i, subdata in enumerate(self.data):
      self.xi[0].append(self.polarity*subdata)
      self.xi[1].append(np.tile(i, len(subdata)))
    self.xi[0] = np.hstack(self.xi[0])    
    self.xi[1] = np.hstack(self.xi[1])    

#-------------------------------------------------------------------------------
  def dgei(self, nres=None, qres=128, gres=128, pgb=None):
    nres = nres or self.nlims[1]
    nset = np.array(np.round(np.linspace(self.nlims[0], self.nlims[1], nres)),
                    dtype=int).tolist()
    self.n = pb.RV('n', nset, pscale='log')
    self.n.set_prob(lambda x: 1./x)
    self.q = pb.RV('q', self.qlims, pscale='log')
    self.g = pb.RV('g', self.glims, pscale='log')
    self.q.set_ufun((np.log, np.exp,))
    self.g.set_ufun((np.log, np.exp,))
    self.paras = pb.RF(self.n, self.q, self.g)
    self.model = pb.SD(self.stats, self.paras)
    self.model.set_prob(lqlf, eps=self.eps, pgb=pgb)
    call = {'x,i': self.xi, 'n':np.array(nset), 'q': {qres}, 'g': {gres}}
    ll = self.model(call)
    self.llx = ll.rescaled(np.complex(np.mean(ll.prob)))
    I = np.ravel(self.llx['i'])
    u = np.unique(I).tolist()
    self.lli = [self.llx({'i':i}, keepdims=True) for i in u]
    self.ll = [ll.prod(['x', 'i']) for ll in self.lli]
    ll = tuple(self.ll)
    self.likelihood = pb.product(*ll)
    vals = self.likelihood('cond')
    self.prior = self.paras(vals)
    return self.set_joint(self.prior * self.likelihood)

#-------------------------------------------------------------------------------
  def set_joint(self, joint=None):
    self.joint = joint
    if joint is None:
      return None
    self.posterior = self.joint.conditionalise(['x', 'i'])
    self.post = self.posterior.rescaled()
    self.marg_nq = self.post.marginalise('g')
    self.marg_ng = self.post.marginalise('q')
    self.marg_qg = self.post.marginalise('n')
    self.marg_n = self.marg_nq.marginal('n')
    self.marg_q = self.marg_nq.marginal('q')
    self.marg_g = self.marg_qg.marginal('g')
    self.hatn = self.marg_n.quantile()['n']
    self.hatq = self.marg_q.quantile()['q']
    self.hatg = self.marg_g.quantile()['g']
    self.hatv = 1. / np.sqrt(self.hatg)
    self.hatl = self.hatq / self.hatg
    hatp = self.polarity * self.means / (self.hatq * self.hatn)
    self.hatp = np.maximum(np.minimum(hatp, np.max(self.plims)), np.min(self.plims))
    self.hate = self.eps
    return self.post

#-------------------------------------------------------------------------------
