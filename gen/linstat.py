import numpy as np
import scipy.stats as stats

def multregr(_X, _y, a = 0.05, opts = None):
  # function [hatb, bci, haty, hatM, covM, hats2, t, rsq, F, pF, r] = multregr(X, y, a);
  #
  # Performs multivariate regression of input data X for output vector y, assuming a linear
  # relationship y = Xb, giving confidence limits and test statistics for alpha a (Default 0.05).
  #
  # Outputs (The number of which can be specified by opts, default = all):
  # 
  # hatb = estimated parameters for b (default)
  # bci = confidence interval using alpha a (default 0.05);
  # haty = estimated y
  # hatM = hat matrix (where haty = hatM * y)
  # covM = covariance matrix for b [ = inv (X' * X) ]
  # hats2 = estimated residual variance
  # t = t-value used to calculate the confidence intervals at 1-alpha
  # rsq = r-squared statistic
  # F = F statistic.
  # pF = P-value for F.
  # r = residual differences (not absolute).  
  
  X, y = np.matrix(_X, dtype = float), np.matrix(np.ravel(_y), dtype = float).T
  
  # Use X to calculate degrees of freedom 
  
  N, n = X.shape
  dof = (float)(N-n);
  
  # Calculate covariances and product with transposed X
  
  covM = np.linalg.inv(X.T*X); 
  covMXT = covM * X.T;
  
  # Calculate least squares solution, hat matrix, and estimated outputs
  
  hatb = covMXT * y;
  hatM = X*covMXT;
  haty = hatM * y;
  
  # Calculate estimated variance of residuals
  
  r = y - haty;
  rr = r.T * r;
  hats2 = rr / (dof+1e-300);
  
  # Calculate confidence intervals based on s.d. estimate & co-std.dev.
  
  hats = np.sqrt(hats2);
  cosd = np.sqrt(np.diag(covM));
  t = stats.t.ppf((1.-a/2 ), dof); 
  if np.isnan(t): t = 0.
  bci = np.tile(hatb, [1, 2]) + (hats*cosd).T * np.matrix([-t, t], dtype = float);
  
  # Calculate r-squared statistic
  
  my = np.mean(y);
  ssq = np.sum( np.power(y - my, 2) );
  rssq = np.sum( np.power(haty - my, 2) );
  rsq = rssq / ssq;
  
  # Calculate F-statistic and P-value
  
  F = rssq / (hats2 * (n-1.));
  pF = 1. - stats.f.cdf(F, n-1., dof);
 
  argsout = hatb, bci, haty, hatM, covM, hats2, t, rsq, F, pF, r
  if opts is None: return argsout
  return argsout[:(opts+1)]
