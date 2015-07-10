"""
    MTMSpectrum
    -----------
    Contains class implementing Thompson's multi-taper method with line detection and 
    reshaping.    
"""

__author__ = 'cwolfe'
from numpy import *
import numpy.matrixlib as ma
import scipy as sp

class MTMSpectrum:
    """
    Class implementing Thompson's multi-taper method with line detection, reshaping, and 
    reconstruction
    
    Attributes
    ----------
    data : array_like
        Input time series
    
    Parameters
    ----------
    nw : int
        Time-bandwidth product
    K : int
        number of tapers to use
    """
#    global np
    
    def __init__(self, data, nw, K, t0=0, fs=None, dt=None, nfft=None, adaptTol=None, adaptMaxIts=20,
                 useEffectiveDOF=True):

        self.data = data.squeeze()
        self.nw = nw
        self.K = K
        self.useEffectiveDOF = useEffectiveDOF
        self.Hk = None
        self.S_noiseModel = None
        self.t0 = t0

        self.dpss = None
        self.lk = None
        self.f = None
        self.Nfreq = None
        self.df = None
        self.fn = None
        self.fr = None
        self.Jk = None
        self.Sk = None
        self.S_noise = None
        self.dof = None
        self.S = None
        self.bk = None
        self.fitType = None
        self.noiseCoef = None
        self.S_smoothed = None
        self.S_noiseModel = None

        self.confidenceValue = None
        self.confidenceInterval = None
        self.confidenceLimit = None

        self.S_line = None
        self.Fstat = None
        self.lines = None
        self.fLines = None

        self.removedLines = None
        self.fRemovedLines = None

        # import ipdb; ipdb.set_trace()

        if fs is None:
            if dt is not None:
                self.dt = float(dt)
                self.fs = 1 / self.dt
            else:
                self.dt = 1.
                self.fs = 1.
        else:
            if dt is None:
                self.fs = float(fs)
                self.dt = 1 / self.fs
            else:
                raise ValueError("Can't specify both fs and dt")

        self.N = self.data.size

        if nfft is None:
            self.nfft = int(2 ** (ceil(log2(self.N))))  # next power of two
            if self.nfft < 1.5 * self.N: self.nfft *= 2
        else:
            self.nfft = nfft

        # finished with setup, calculate the spectrum
        self.calcSpectrum(adaptTol, adaptMaxIts)

    def calcSpectrum(self, tol=None, MaxIts=20):
        """ 
        Calculate the spectrum.
        
        Tapers are optimally combined using adaptive weighting. See :func:`adaptWeights`
        for an explanation of input variables.
        
        Called automatically on construction.        
        """
        from nitime.algorithms.spectral import dpss_windows

        self.dpss, lam = dpss_windows(self.N, self.nw, self.K)
        self.lk = lam[:, newaxis]

        self.f = fft.rfftfreq(self.nfft, d=self.dt)
        self.Nfreq = self.f.size
        self.df = self.f[1] - self.f[0]
        self.fn = self.f[-1]
        self.fr = self.fs / self.N

        self.Jk = fft.rfft(self.dpss * self.data, self.nfft, 1)
        self.Sk = abs(self.Jk) ** 2

        self.adaptWeights(tol, MaxIts)
        self.S_noise = self.S.copy()

    def adaptWeights(self, tol=None, MaxIts=20):
        """ 
        Perform the adaptive weighting.
        
        Parameters
        ----------
        tol : float, optional
            Stopping criterion for adaptive weighting iteration. If `None` defaults to
            0.005 times the signal variance divided by the FFT length.
        MaxIts : int, optional
            Maximum number of iterations to perform before quiting.
        """
        
        import warnings
        sig2 = self.data.var()  # power

        Sk = self.Sk
        lk = self.lk

        if tol is None: tol = 0.005 * sig2 / self.nfft

        S = (Sk[0, :] + Sk[1, :]) / 2  # initial esimate of spectrum
        a = sig2 * (1 - lk)

        for n in range(MaxIts):
            b = S / (lk * S + a)
            S1 = (lk * b ** 2 * Sk).sum(axis=0) / (lk * b ** 2).sum(axis=0)

            if abs(S - S1).mean() < tol: break

            Stemp = S1  # swap S and S1
            S1 = S
            S = Stemp
        else:
            warnings.warn('Adaptive weight iteration did not converge')

        # effective degrees of freedom
        if self.useEffectiveDOF:
            self.dof = 2 * sum(b ** 2 * lk, axis=0) ** 2 / sum(b ** 4 * lk ** 2, axis=0)
        else:
            self.dof = tile(2 * self.K, (self.Nfreq))

        self.S = S
        self.bk = b

    def ar1(self, f, sig2, r):
        r"""
        The spectrum of an AR(1) process.
        
        Notes
        -----
        The spectrum of an AR(1) process with variance :math:`\sigma^2` and lag-1 autocorrelation
        :math:`r` is
        
        .. math::
            S(f) = \sigma^2\frac{1 - r^2}{1 + r^2 - 2r\cos\pi f},
            
        where :math:`f` is normalized by the Nyquist frequency.
        """
        return sig2 * (1 - r ** 2) / (1 + r ** 2 - 2 * r * cos(pi * f))

    def lorentzian(self, f, sig2, gamma):
        r"""
        A Lorentzian centered at zero
        
        Notes
        -----
        The Lorentzian is given by
        
        .. math::
            L(f) = \frac{2}{\pi\gamma}\frac{\sigma^2}{1 + \frac{f^2}{\gamma^2}},
            
        where :math:`\gamma` and :math:`\sigma^2` are the decorrelation time and variance, respectively, of
        the time series.
        """ 
        return (2. / (pi * gamma)) * sig2 / (1 + f ** 2 / gamma ** 2)

    def getConfidence(self, confidenceValue, smoothingWidth, fitType='ar1', fitRange=None):
        """
        Get confidence intervals.
        
        Confidence intervals are based on a noise model. The noise model is a fit in 
        spectral space to a median smoothed version of the noise spectrum.
        """
        from scipy.optimize import curve_fit
        from scipy.stats import chi2

        f = self.f
        S = self.S_noise
        dof = self.dof

        if smoothingWidth is None:
            raise ValueError('Must specify smoothingWidth')

        S_smoothed = medianSmooth(S, smoothingWidth)

        if fitRange is None:
            idx = f <= self.fn
        else:
            idx = logical_and(f >= fitRange[0], f <= fitRange[1])

        Ssvar = S_smoothed[idx].sum() * self.df

        if fitType == 'ar1':
            self.fitType = 'ar1'
            popt, pcov = curve_fit(self.ar1, f[idx] / self.fn, S_smoothed[idx], [Ssvar, .5])
            S_noiseModel = self.ar1(f / self.fn, popt[0], popt[1])
        elif fitType == 'lorentzian':
            self.fitType = 'lorentzian'
            popt, pcov = curve_fit(self.lorentzian, f[idx], S_smoothed[idx], [Ssvar, .05])
            S_noiseModel = self.lorentzian(f, popt[0], popt[1])
        else:
            raise ValueError('Unknown fitType: possible values are "ar1" and "lorentzian"')

        self.noiseCoef = popt
        self.S_smoothed = S_smoothed
        self.S_noiseModel = S_noiseModel

        self.confidenceValue = confidenceValue
        self.confidenceInterval = dof / chi2.ppf((1 - confidenceValue) / 2, dof)
        self.confidenceLimit = self.confidenceInterval * S_noiseModel

    def findLines(self, chiTestCutoff, fTestCutoff, smoothingWidth=None, fitType='ar1', fitRange=None):
        r"""
        Find lines.
        
        Uses isolated peak detection with the F-test and significant power
        above the noise model.
        
        If the noise model has not been instantiated, calls :func:`getConfidence` first. See
        this method for a description of `smoothingWidth`, `fitType`, and `fitRange`.
        
        Parameters
        ----------
        chiTestCutoff : float
            Amount by which the chi-squared statistic of the line must exceed the noise model.
        fTestCutoff : float
            Cutoff for F-statistic.
        
        Notes
        -----
        We follow Percival and Walden and esimate the coefficients of the lines as
        
        .. math:: 
            \hat{C}_l = \frac{\sum_{k\text{ even}}^{K-1} J_{kl} H_k(0)}{\sum_{k\text{ even}}^{K-1} H_k(0)^2},
            
        where
        
        .. math::
            J_{kl} = \sum_{n=1}^N h_{nk} X_n e^{-2\pi i l n/N}, \quad\quad H_{kl} = \sum_{n=1}^N h_{nk}e^{-2\pi i l n/N},
            
        and the :math:`h_{nk}` are the DPSS tapers.
        
        The test statistic is 
        
        .. math::
            F_l = \frac{(K-1)|\hat{C}_l|^2 \sum_{k=0}^{K-1}H_k^2(0)}{\sum_{k=0}^{K-1} |J_{kl} - H_k(0)\hat{C}_l|^2}.
        """
        from scipy.stats import f as fdist
        from scipy.stats import chi2

        if self.S_noiseModel is None:
            self.getConfidence(chiTestCutoff, smoothingWidth, fitType, fitRange)

        S = self.S
        h = self.dpss
        K = self.K
        Jk = self.Jk
        dof = self.dof
        nfft = self.nfft
        S_noiseModel = self.S_noiseModel

        if self.Hk is None: self.Hk = fft.rfft(h, nfft, 1)
        Hk = self.Hk

        Hk0 = real(Hk[:, 0][:, newaxis])
        Cl = (sum(Jk * Hk0, axis=0) / sum(Hk0 ** 2))[newaxis, :]

        F = squeeze((K - 1) * abs(Cl) ** 2 * sum(Hk0 ** 2) / sum(abs(Jk - Hk0 * Cl) ** 2, axis=0))

        lines = nonzero(
            logical_and(
                S > dof / chi2.ppf((1 - chiTestCutoff) / 2, dof) * S_noiseModel,
                F > fdist.ppf(fTestCutoff, 2, 2 * K - 2)))[0]

        self.S_line = Cl
        self.Fstat = F
        self.lines = lines
        self.fLines = self.f[lines]

    def removeLines(self, linesToRemove, lineWidth=None):
        r"""
        Remove given lines; reshape spectrum.
        
        Parameters
        ----------
        linesToRemove : list of int
            Indices of lines to remove.
        lineWidth : int
            Half-width of the reshaping area around the line (default is the smallest integer
            greater than `nw`).
            
        Notes
        -----
        Near a line located at :math:`l = l_0`, the reshaped noise spectrum is
        
        .. math::
            \hat{S}_l = \frac{1}{K}\sum_{k=0}^{K-1}|J_{kl} - \hat{C}_{l_0}H_{k,l-l0}|^2.
        """
        #    import ipdb; ipdb.set_trace()
        S = self.S
        Hk = self.Hk
        Jk = self.Jk
        nw = self.nw
        Cl = self.S_line

        lines = asarray(linesToRemove)

        if lineWidth is None:
            width = tile(int(ceil(nw)), lines.shape)
        else:
            width = asarray(lineWidth, dtype=int) - 1

        Shat = self.S_noise

        for l0, W in zip(lines, width):
            l = range(max(l0 - W, 0), min(l0 + W + 1, S.size))
            Hloc = Hk[:, abs(l - l0)]
            Hloc[:, l < l0] = conj(Hloc[:, l < l0])
            Shat[l] = mean(abs(Jk[:, l] - Cl[0, l0] * Hloc) ** 2, axis=0)

        self.removedLines = lines
        self.fRemovedLines = self.f[lines]

    def resetLines(self):
        """
        Clear the removed lines. Reset noise spectrum.
        """
        self.removedLines = None
        self.fRemovedLines = None
        self.S_noise = self.S.copy()

    def getEnvelope(self, line, laplaceNeumann=True, del4BC=(2,3), tau=(0.,0.), data=None):
        r"""
        Signal reconstruction.
        
        Parameters
        ----------
        line : int
            Index of line to reconstruct
        laplaceNeumann : bool, optional
            Whether to use Neumann or Dirichlet boundary conditions on the Laplacian inversion 
            (default is true).
        del4BC : {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}, optional
            Which derivatives to set to zero for the biharmonic inversion (default is (2,3), 
            i.e., the 2nd and 3rd derivatives).
        tau : tuple of two floats
            Smoothing timescales for the Laplacian and biharmonic operators (default is (0,0)).
        data : array_like
            Time series used to compute the misfit of the signal reconstruction.
            
        Returns
        -------
        a, x : array_like
            Complex envelope and real signal, respectively, of the reconstructed signal.
        rmse : float
            RMS error from `data` of the reconstructed signal.
        J0, J2, J4 : float
            Contributions to the cost function from the 0th, 2nd, and 4th order terms, respectively.
            
        Notes
        -----
        The envelope :math:`a` of the reconstructed signal minimizes the cost function
        
        .. math::
            \mathcal{J} = |\mathbf{a}|^2 + \tau_1^2 |\dot{\mathbf{a}}|^2 + \tau_2^4|\ddot{\mathbf{a}}|^2,
            
        where the dots indicate time differentiation, subject to the constraints
        
        .. math::
            J_k(f_0) = \mathbf{g}_k^\mathrm{T} \mathbf{a},
            
        where :math:`\mathbf{g}_k` is the :math:`k^\text{th}` DPSS taper. Define :math:`\mathrm{\mathbf{D}}_2`
        and :math:`\mathrm{\mathbf{D}}_4` as the discrete second and fourth derivative matrices, respectively,
        with appropriate boundary conditions, and let
        
        .. math::
            \mathrm{\mathbf{H}} = \mathrm{\mathbf{I}} + \tau_1^2 \mathrm{\mathbf{D}}_2 + \tau_2^4 \mathrm{\mathbf{D}}_4.
    
        The solution can then be written as
        
        .. math::
            \mathbf{a} = \mathrm{\mathbf{H}}^{-1}\mathrm{\mathbf{G}}
                \left(\mathrm{\mathbf{G}}\mathrm{\mathbf{H}}^{-1}\mathrm{\mathbf{G}}^\mathrm{T}\right)^{-1}
                \mathbf{y},
                           
        where :math:`\mathbf{y}` is a vector of the :math:`J_k(f_0)\text{s}`.         
        """
        from numpy.linalg import norm
    #    import ipdb; ipdb.set_trace()

        f0 = self.f[line]
        N = self.N
        t = arange(N) * self.dt

        # factor to account for the fact that we only use positive frequencies:
        if f0 > 0:
            freqFac = 2.0
        else:
            freqFac = 1.0    

        y = freqFac*ma.asmatrix(self.Jk[:, line]).T

        G = ma.asmatrix(self.dpss)
        I = sp.sparse.identity(N, format='csr')
        D2 = sp.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format="csr")
        D4 = D2.T*D2
    
    
        # boundary conditions
        if laplaceNeumann:
            D2[0,0] = -1
            D2[-1,-1] = -1

        
        if del4BC == (0,1): # 0th and first derivative zero
            D4[0,:3]   = [6, -4, 1]
            D4[-1,-3:] = [1, -4, 6]
        
            D4[1,:4]   = [-4,  6, -4,  1]
            D4[-2,-4:] = [ 1, -4,  6, -4]
        elif del4BC == (0,2): # 0th and 2nd derivative zero
            D4[0,:3]   = [5, -4, 1]
            D4[-1,-3:] = [1, -4, 5]
        
            D4[1,:4]   = [-4,  6, -4,  1]
            D4[-2,-4:] = [ 1, -4,  6, -4]
        elif del4BC == (0,3): # 0th and 3rd derivative zero
            D4[0,:3]   = [3, -3, 1]
            D4[-1,-3:] = [1, -3, 3]
        
            D4[1,:4]   = [-4,  6, -4,  1]
            D4[-2,-4:] = [ 1, -4,  6, -4]
        elif del4BC == (1,2): # 1st and 2nd derivative zero
            D4[0,:3]   = [3, -4, 1]
            D4[-1,-3:] = [1, -4, 3]
        
            D4[1,:4]   = [-3,  6, -4,  1]
            D4[-2,-4:] = [ 1, -4,  6, -3]
        elif del4BC == (1,3): # 1st and 3rd derivative zero
            D4[0,:3]   = [2, -3, 1]
            D4[-1,-3:] = [1, -3, 2]
        
            D4[1,:4]   = [-3,  6, -4,  1]
            D4[-2,-4:] = [ 1, -4,  6, -3]
        elif del4BC == (2,3): # 2st and 3rd derivative zero
            D4[0,:3]   = [1, -2, 1]
            D4[-1,-3:] = [1, -2, 1]
        
            D4[1,:4]   = [-2,  5, -4,  1]
            D4[-2,-4:] = [ 1, -4,  5, -2]
        else:
            raise ValueError('Impossible combination of boundary conditions')
    
        H0 = I
        H2 = D2*(tau[0]/self.dt)**2
        H4 = D4*(tau[1]/self.dt)**4

        H = H0 + H2 + H4

        Gamma = G * sp.sparse.linalg.spsolve(H, G.T)
        Gi = linalg.inv(Gamma)

        a = sp.sparse.linalg.spsolve(H.astype(complex), G.T * Gi * y)
        x = real(a * exp(2j * pi * f0 * t))

        if data is None:
            data = self.data

        rmse = sum((x - data) ** 2)

        return a, x, rmse, norm(H0 * ma.asmatrix(a).T), norm(H2 * ma.asmatrix(a).T), norm(H4 * ma.asmatrix(a).T)


def medianSmooth(x,width):
    """
    Median smoothing without zero padding.
    
    Edges are handled by shrinking the width of the smoother.
    """
    w = 2*floor(1+width/2)-1  # ensure width is an odd integer
    nw = (w-1)/2

    N = len(x)

    y = zeros(x.shape)
    for n in range(N):
        ix0 = max(n-nw,1)
        ixf = min(n+nw,N)

        y[n] = median(x[ix0:ixf])

    return y
