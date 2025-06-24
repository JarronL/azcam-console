"""
Small collection of robust statistical estimators based on functions from
Henry Freudenriech (Hughes STX) statistics library (called ROBLIB) that have
been incorporated into the AstroIDL User's Library.  

Updated by J. Leisenring (UA) to be more pythonic.

Function included are:

  * medabsdev - median absolute deviation
  * mean - robust estimator of the mean of a data set
  * mode - robust estimate of the mode of a data set using the half-sample
    method
  * std - robust estimator of the standard deviation of a data set

"""

from __future__ import division

import numpy as np
from numpy import median, nanmedian

import logging
_log = logging.getLogger('webbpsf_ext')

__version__ = '0.4'
__revision__ = '$Rev$'
__all__ = ['median', 'nanmedian', 'mean', 'mode', \
           'medabsdev','biweightMean', 'std', \
           'checkfit', 'linefit', 'polyfit', \
           '__version__', '__revision__', '__all__']


# Numerical precision
__epsilon = np.finfo(float).eps

def medabsdev(data, axis=None, keepdims=False, nan=True):
    """Median Absolute Deviation
    
    A "robust" version of standard deviation. Runtime is the 
    same as `astropy.stats.funcs.mad_std`.
    
    Parameters
    ----------
    data : ndarray
        The input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the deviation is computed. The
        default is to compute the deviation of the flattened array.
        
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
        This is the equivalent of reshaping the input data and then taking
        the standard devation.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
    nan : bool, optional
        Ignore NaNs? Default is True.
    """
    medfunc = np.nanmedian if nan else np.median
    meanfunc = np.nanmean if nan else np.mean

    if (axis is None) and (keepdims==False):
        data = data.ravel()
    
    # Scale factor to return result equivalent to standard deviation.
    sig_scale = 0.6744897501960817
    
    med = medfunc(data, axis=axis, keepdims=True)
    absdiff = np.abs(data - med)
    sigma = medfunc(absdiff, axis=axis, keepdims=True)  / sig_scale
    
    # Check if anything is near 0.0 (below machine precision)
    mask = sigma < __epsilon
    if np.any(mask):
        sigma[mask] = (meanfunc(absdiff, axis=axis, keepdims=True))[mask] / 0.8
    mask = sigma < __epsilon
    if np.any(mask):
        sigma[mask] = 0.0
        
        
    if len(sigma)==1:
        return sigma[0]
    elif not keepdims:
        return np.squeeze(sigma)
    else:
        return sigma



def mean(inputData, Cut=3.0, axis=None, dtype=None, keepdims=False, 
    return_std=False, return_mask=False):
    """Robust Mean
    
    Robust estimator of the mean of a data set. Based on the `resistant_mean` 
    function from the AstroIDL User's Library. NaN values are excluded.

    This function trims away outliers using the median and the median 
    absolute deviation. An approximation formula is used to correct for
    the truncation caused by trimming away outliers.

    Parameters
    ==========
    inputData : ndarray
        The input data.

    Keyword Args
    ============
    Cut : float
        Sigma for rejection; default is 3.0.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the deviation is computed. The
        default is to compute the deviation of the flattened array.
        
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
        This is the equivalent of reshaping the input data and then taking
        the standard devation.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
    return_std : bool
        Also return the std dev calculated using only the "good" data?
    return_mask : bool
        If set to True, then return only boolean array of good (1) and 
        rejected (0) values.

    """

    inputData = np.array(inputData)
    
    if np.isnan(inputData).sum() > 0:
        medfunc = np.nanmedian
        meanfunc = np.nanmean
    else:
        medfunc = np.median
        meanfunc = np.mean

    if axis is None:
        data = inputData.ravel()
    else:
        data = inputData
        
    if type(data).__name__ == "MaskedArray":
        data = data.compressed()
    if dtype is not None:
        data = data.astype(dtype)

    # Scale factor to return result equivalent to standard deviation.
    sig_scale = 0.6744897501960817
        
    # Calculate the median absolute deviation
    data0 = medfunc(data, axis=axis, keepdims=True)
    absdiff = np.abs(data-data0)
    medAbsDev = medfunc(absdiff, axis=axis, keepdims=True) / sig_scale

    mask = medAbsDev < __epsilon
    if np.any(mask):
        medAbsDev[mask] = (meanfunc(absdiff, axis=axis, keepdims=True))[mask] / 0.8

    # First cut using the median absolute deviation
    cutOff = Cut*medAbsDev
    good = absdiff <= cutOff
    data_naned = data.copy()
    data_naned[~good] = np.nan
    dataMean = np.nanmean(data_naned, axis=axis, keepdims=True)
    dataSigma = np.nanstd(data_naned, axis=axis, keepdims=True)
    #dataSigma = np.sqrt( np.nansum((data_naned-dataMean)**2.0) / len(good) )

    # Calculate sigma
    if Cut > 1.0:
        sigmaCut = Cut
    else:
        sigmaCut = 1.0
    if sigmaCut <= 4.5:
        poly_sigcut = -0.15405 + 0.90723*sigmaCut - 0.23584*sigmaCut**2.0 + 0.020142*sigmaCut**3.0
        dataSigma = dataSigma / poly_sigcut

    cutOff = Cut*dataSigma
    good = absdiff <= cutOff

    if return_mask:
        return np.reshape(~np.isnan(data_naned), inputData.shape)

    data_naned = data.copy()
    data_naned[~good] = np.nan
    dataMean = np.nanmean(data_naned, axis=axis, keepdims=True)
    if return_std:
        dataSigma = np.nanstd(data_naned, axis=axis, keepdims=True)
    
    if len(dataMean)==1:
        if return_std:
            return dataMean[0], dataSigma[0]
        else:
            return dataMean[0]
    if not keepdims:
        if return_std:
            return np.squeeze(dataMean), np.squeeze(dataSigma)
        else:
            return np.squeeze(dataMean)
    else:
        if return_std:
            return dataMean, dataSigma
        else:
            return dataMean



def mode(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:
        fnc = lambda x: mode(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                wMin = data[-1] - data[0]
                N = int(data.size/2 + data.size%2)
                for i in range(0, N):
                    w = data[i+N-1] - data[i] 
                    if w < wMin:
                        wMin = w
                        j = i
                return _hsm(data[j:j+N])
            
        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)
        
        # The data need to be sorted for this to work
        data = np.sort(data)
    
        # Find the mode
        dataMode = _hsm(data)
    
    return dataMode

def std(inputData, Zero=False, axis=None, dtype=None, 
        keepdims=False, return_mask=False, ddof=1.0):
    """Robust Sigma
    
    Based on the robust_sigma function from the AstroIDL User's Library.

    Calculate a resistant estimate of the dispersion of a distribution.
    
    Use the median absolute deviation as the initial estimate, then weight 
    points using Tukey's Biweight. See, for example, "Understanding Robust
    and Exploratory Data Analysis," by Hoaglin, Mosteller and Tukey, John
    Wiley & Sons, 1983, or equation 9 in Beers et al. (1990, AJ, 100, 32).

    Parameters
    ==========
    inputData : ndarray
        The input data.

    Keyword Args
    ============
    axis : None or int or tuple of ints, optional
        Axis or axes along which the deviation is computed. The
        default is to compute the deviation of the flattened array.
        
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
        This is the equivalent of reshaping the input data and then taking
        the standard devation.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
    return_mask : bool
        If set to True, then only return boolean array of good (1) and 
        rejected (0) values.
    ddof : int
    Delta Degrees of Freedom. The divisor used in calculations is N - ddof, 
    where N represents the number of elements. By default ddof is 1. This 
    differs from numpy.std which uses ddof=0 by default.
	"""

    inputData = np.array(inputData)
    
    if np.isnan(inputData).sum() > 0:
        medfunc = np.nanmedian
        meanfunc = np.nanmean
    else:
        medfunc = np.median
        meanfunc = np.mean

    if axis is None:
        data = inputData.ravel()
    else:
        data = inputData

    if type(data).__name__ == "MaskedArray":
        data = data.compressed()
    if dtype is not None:
        data = data.astype(dtype)

    # Scale factor to return result equivalent to standard deviation.
    sig_scale = 0.6744897501960817

    # Calculate the median absolute deviation
    if Zero:
        data0 = 0.0
    else:
        data0 = medfunc(data, axis=axis, keepdims=True)
    absdiff = np.abs(data-data0)
    medAbsDev = medfunc(absdiff, axis=axis, keepdims=True) / sig_scale
    mask = medAbsDev < __epsilon
    if np.any(mask):
        medAbsDev[mask] = (meanfunc(absdiff, axis=axis, keepdims=True))[mask] / 0.8
        
    # These will be set to 0 later
    mask0 = medAbsDev < __epsilon
        
    u = (data-data0) / (6.0 * medAbsDev)
    u2 = u**2.0
    good = u2 <= 1.0

    if return_mask:
        return good & ~np.isnan(data)
    
    # These values will be set to NaN later
    # if fewer than 3 good points to calculate stdev
    ngood = good.sum(axis=axis, keepdims=True)
    mask_nan = ngood < 2
    if mask_nan.sum() > 0:
        _log.warning("NaN's will be present due to weird distributions")
    
    # Set bad points to NaNs
    u2[~good] = np.nan
    
    numerator = np.nansum( (data - data0)**2 * (1.0 - u2)**4.0, axis=axis, keepdims=True)
    nElements = len(data) if axis is None else data.shape[axis]
    denominator = np.nansum( (1.0 - u2) * (1.0 - 5.0*u2), axis=axis, keepdims=True)
    sigma = np.sqrt( nElements*numerator / (denominator*(denominator-ddof)) )
    
    sigma[mask0] = 0
    sigma[mask_nan] = np.nan

    if len(sigma)==1:
        return sigma[0]
    elif not keepdims:
        return np.squeeze(sigma)
    else:
        return sigma

