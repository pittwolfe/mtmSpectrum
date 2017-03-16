"""
    MTMSpectrum
    -----------
    Contains class implementing Thompson's multi-taper method with line detection and 
    reshaping.    
"""

__author__ = 'cwolfe'

import numpy as np
import numpy.matrixlib as ma
import scipy as sp

class MTMSpectrum(object):
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
    
    def __init__(self, data, time_bandwidth=None, t0=0, dt=None, fs=None, number_of_tapers=None, 
                 nfft=None, adaptTol=None, adaptMaxIts=1000, useEffectiveDOF=True, calcSpectrum=True):

        from .util import dpss
        
        if isinstance(data, MTMSpectrum):
            obj = data
            
            self.data               = obj.data
            self.xvar               = obj.xvar
            self.time_bandwidth     = obj.time_bandwidth
            self.number_of_tapers   = obj.number_of_tapers
            self.useEffectiveDOF    = obj.useEffectiveDOF
            self.fs                 = obj.fs
            self.dt                 = obj.dt
            self.N                  = obj.N
            self.nfft               = obj.nfft
            self.adaptTol           = obj.adaptTol
            self.adaptMaxIts        = obj.adaptMaxIts
            self.fr                 = obj.fr
            self.fn                 = obj.fn
            self.Nfreq              = obj.Nfreq
            self.df                 = obj.df
            self.freq               = obj.freq
            self.dpss               = obj.dpss
            self.dpss_concentration = obj.dpss_concentration
            self.spec               = obj.spec
            self.edof               = obj.edof
            self.dof                = obj.dof
            self.eigenFT            = obj.eigenFT
            self.weights            = obj.weights
        else:
                    
            self.data = data.squeeze()
            self.xvar = data.var(ddof=1) # variance of original data
        
            self.time_bandwidth = time_bandwidth
            self.useEffectiveDOF = useEffectiveDOF
        
            if number_of_tapers is None:
                self.number_of_tapers = np.ceil(2*time_bandwidth) - 1
            else:
                self.number_of_tapers = number_of_tapers

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
                self.nfft = int(2 ** (np.ceil(np.log2(self.N))))  # next power of two
                if self.nfft < 1.5 * self.N: self.nfft *= 2
            else:
                self.nfft = nfft
            
            if adaptTol is None: 
                self.adaptTol = 0.005 * self.xvar / self.nfft
            else:
                self.adaptTol = adaptTol
            self.adaptMaxIts = adaptMaxIts
        
            self.t0 = t0
                
            self.fr    = 1/(self.N*dt) # Rayleight frequency
            self.fn    = 1/(2*dt) # Nyquist frequency
            self.Nfreq = self.nfft//2 + 1
            self.df    = 1/(self.nfft*dt)
            self.freq  = np.arange(0, self.Nfreq)*self.df
            
            # DOF in the raw spectra
            self.dof = np.ones(self.Nfreq)
            if np.mod(self.nfft, 2) == 0:
                self.dof[1:-1] = 2
            else:
                self.dof[1:] = 2
        
            dpss, self.dpss_concentration = dpss(self.N, self.time_bandwidth, self.number_of_tapers)
            self.dpss = dpss.T
        
            self.calcSpectrum(self.adaptTol, self.adaptMaxIts)
            
    def copy(self):
        """ return a shallow copy of self """
        result = MTMSpectrum(self)
        
        return result
        

    def calcSpectrum(self, tol=None, MaxIts=None):
        """ 
        Calculate the spectrum.
        
        Called automatically on construction.        
        
        This method estimates the adaptive weighted multitaper spectrum, as in
        Thomson 1982.  This is done by estimating the DPSS (discrete prolate
        spheroidal sequences), multiplying each of the tapers with the data series,
        take the FFT, and using the adaptive scheme for a better estimation.

        The spectrum is
            $$S(f) = \frac{\sum_{k=0}^{K-1} b_k^2(f) \mu_k |Y_k(f)|^2}{\sum_{k=0}^{K-1} b_k^2(f) \mu_k},$$
        where the $Y_k(f)$ are the eigen-FTs,
            $$Y_k(f) = \sum_{n=0}^{N-1} w_k(t_n) X(t_n) \mathrm{e}^{-2\pi i f t_n},$$
        $t_n = n\Delta t$ for $n = 0, \dots, N-1$, $w_k$ is the $k^\text{th}$ DPSS, 
        and $\mu_k$ is the $k^\text{th}$ DPSS concentration. The weights $b_k$ are 
        determined by the adaptive weighting proceedure implemented in `adaptspec`.

        :param data: :class:`numpy.ndarray`
             Array with the data.
        :param dt: float
             Sample spacing of the data.
        :param time_bandwidth: float
             Time-bandwidth product. Common values are 2, 3, 4 and numbers in
             between.
        :param nfft: int
             Number of points for fft. If nfft == None, no zero padding
             will be applied before the fft
        :param number_of_tapers: integer, optional
             Number of tapers to use. Defaults to int(2*time_bandwidth) - 1. This
             is maximum senseful amount. More tapers will have no great influence
             on the final spectrum but increase the calculation time. Use fewer
             tapers for a faster calculation.
        :param adaptive: bool, optional
             Whether to use adaptive or constant weighting of the eigenspectra.
             Defaults to True(adaptive).
         
        :return spec: :class:`numpy.ndarray`
            power spectrum estimate
        :return freq: :class:`numpy.ndarray`
            frequencies
        :return nu: :class:`numpy.ndarray`
            effective degrees of freedom
        :return yk: :class:`numpy.ndarray`
            the eigenspectra
        :return wk: :class:`numpy.ndarray`
            The dpss
        :return mu: :class:`numpy.ndarray`
            The dpss concetrations
        """
        
        x    = self.data
        npts = self.N
        nf   = self.Nfreq
        nfft = self.nfft
        wk   = self.dpss
        mu   = self.dpss_concentration
        df   = self.df
        
        self.eigenFT = np.fft.fft(x[:,np.newaxis]*wk, n=nfft, axis=0)  # eigen-FTs
        sk = np.abs(self.eigenFT[:nf,:])**2 # eigenspectra
        sbar = 2*np.mean(sk/mu[np.newaxis,:],axis=1) # mean spectrum
    
        self.adaptspec(self.adaptTol, self.adaptMaxIts)

        # double the power in positive frequencies
        self.spec *= self.dof
    
        # resscale to match original variance
        sscal = np.sum(self.spec*df)
        self.spec *= self.xvar/sscal
        

    def adaptspec(self, adaptTol=None, adaptMaxIts=None):
        """
        subroutine to calculate adaptively weighted power spectrum

        The adaptive weights $b_k$ are defined by the iteration
            $$b_k^{(n+1)}(f) = \frac{\sqrt{\mu_k} S^{(n)}(f)}{\mu_k S^{(n)}(f) + (1-\mu_k)\frac{1}{K}\sum_{k=0}^{K-1} \sigma_k^2},$$
        where
            $$S^{(n)}(f) = \frac{\sum_{k=0}^{K-1} b_k^{(n)2}(f) \mu_k |Y_k(f)|^2}{\sum_{k=0}^{K-1} b_k^{(n)2}(f) \mu_k},$$
        and
            $$\sigma_k^2 = \Delta f\sum_{m=0}^{\hat{N}-1} |Y_k(f_m)|^2,$$
        The number of points used in the FFT is $\hat{N}$ ($\hat{N} \ge N$), $f_m = m \Delta f$ 
        for $m = 0, \ldots, \hat{N}-1$, and $\Delta f = 1/(\hat{N}\Delta t)$. The starting value 
        for the iteration is 
            $$S^{(0)}(f) = \frac{|Y_0(f)|^2 + |Y_1(f)|^2}{2}$$
        and the iteration continues until 
            $$\max_f = \frac{|S^{(n+1)}(f) - S^{(n)}(f)|}{|S^{(n+1)}(f) + S^{(n)}(f)|} < \text{tol}.$$
    
            inputs:

            mu    - DPSS concentrations
            yk    - array containing kspec eigenFTs

        outputs:

            spec - vector containing adaptively weighted spectrum
            nu   - vector containing the number of degrees of freedom
               for the spectral estimate at each frequency.
            bk   - array containing the ne weights for kspec 
                  eigenspectra normalized so that if there is no bias, the
               weights are unity.
        """

        if adaptTol is None:
            adaptTol = self.adaptTol
        if adaptMaxIts is None:
            adaptMaxIts = self.adaptMaxIts

        mu    = self.dpss_concentration
        yk    = self.eigenFT
        nfft  = self.nfft
        kspec = self.number_of_tapers
        ne    = self.Nfreq
        Nfreq = self.Nfreq
#         df    = self.df
    
#         nfft, kspec = yk.shape
#         ne = int(nfft/2 + 1)    
    
        df = 0.5/(ne - 1) # assume unit sampling
        sk = np.abs(yk[:ne,:])**2
    
        varsk = df*(sk[0,:] + np.sum(sk[1:-1,:], axis=0) + sk[-1,:])
        dvar = np.mean(varsk)
        
        Bk = (1 - mu) * dvar
        sqev = np.sqrt(mu)
    
        cerr = 1 # current error
        rerr = 9.5e-7 # a magical mystery number
    
        # begin iterations
        j = 0
        spec = (sk[:,0] + sk[:,1])/2
    
        while cerr > rerr:
            j += 1
            slast = spec
        
            bk = sqev[np.newaxis,:]*spec[:,np.newaxis]/(mu[np.newaxis,:]*spec[:,np.newaxis] + Bk[np.newaxis,:])
            bk[bk > 1] = 1
        
            spec = np.sum(bk**2*sk, axis=1)/np.sum(bk**2, axis=1)
        
            if j >= 1000:
                warnings.warn('adaptive iteration did not converge')
                break
            
            cerr = np.amax(np.abs((spec - slast)/(spec + slast)))
        
        bk_dofs = bk/(np.sqrt(np.mean(bk**2, axis=1)))[:,np.newaxis]
        bk_dofs[bk_dofs > 1] = 1
    
        nu = self.dof*np.sum(bk_dofs**2, axis=1)
    
        self.spec = spec
        self.edof = nu
        self.weights = bk    
    
    def ar1_spec(self, f, r):
        r"""
        The spectrum of an AR(1) process.
        
        We will try to fit the smoothed spectrum to a AR(1) process that has the spectrum 
            $$S(f) = \frac{\sigma^2}{1 - 2r \cos \pi f/f_\text{N} + r^2},$$
        where $r$ is the lag-one autocorrelation and $\sigma^2$ is the noise variance. 
        The average power in this spectrum is
            $$S_0 = \frac{\sigma^2}{1-r^2},$$
        so we can equivalently write the spectrum as
            $$S(f) = S_0\frac{1-r^2}{1 - 2r \cos \pi f/f_\text{N} + r^2},$$
    
        Notes
        -----
        The spectrum of an AR(1) process with variance and lag-1 autocorrelation
        :math:`r` is
        
        .. math::
            S(f) = S_0\frac{1 - r^2}{1 + r^2 - 2r\cos\pi f/f_n}
        """
        fn = self.fn
        
        return (1-r**2)/(1 - 2*r*np.cos(np.pi*f/fn) + r**2)/fn

    def smoothed_ar1_spec(self, f, fn, r, S0, tau):
        '''
        AR(1) spectrum with Nyquist frequency fn, lag-one autocorrelation r, and total power S0,
        smoothed by a Gaussian with e^2 folding scale of tau^2
        '''
    
        y = np.exp(-2*np.pi**2*f**2*tau**2)*(1-r**2)/(1 - 2*r*np.cos(np.pi*f/fn) + r**2)/fn
        y /= np.sum(y)
    
        return S0*y
        
    def lorentzian_spec(self, f, gamma, L0=1):
        """
        A Lorentzian centered at zero
        
        The unit power Lorentzian is given by
            $$L(f) = \frac{4\gamma}{\gamma^2 + (2\pi f)^2},$$
        where $\gamma$ is the decorrelation time of the time series.        
        
        Notes
        -----
        The Lorentzian is given by
        
        """ 
        return L0*4*gamma/(gamma**2 + (2*np.pi*f)**2)


    def getSmoothedSpectrum(self, smoothingWidth, smoothing_range=None):
        """
        Smooth spectrum with a median filter
        """
        
        f  = self.freq
        df = self.df
        
        spec_smooth = self.copy()
                                  
        spec_smooth.spec = np.full_like(self.spec, np.nan)
        if smoothing_range is None:
            idx = f <= self.fn
        else:
            idx = np.logical_and(f >= smoothing_range[0], f <= smoothing_range[1])
            
        # find the width in terms of bins, ensuring it is an odd integer
        w = 2*int((smoothingWidth/df)//2) + 1
        width = (w-1)//2
        spec_smooth.spec[idx] = medianSmooth(self.spec[idx], width)
        
        return spec_smooth
        
    
    def getFit(self, fitRange=None):
        """ Get a fit to an AR(1) spectrum """
        
        from scipy.optimize import curve_fit
                
        if fitRange is None:
            idx = self.freq <= self.fn
        else:
            idx = np.logical_and(self.freq >= fitRange[0], self.freq <= fitRange[1])
        
        # fit an AR(1) process to it
        popt, _ = curve_fit(lambda f, r, S0: S0*self.ar1_spec(f, r), 
                            self.freq[idx], self.spec[idx], p0=[.5, 1])
        r_opt, S_opt = popt
        print('optimal parameters: r = {:f}, S_0 = {:f}'.format(r_opt, S_opt))
        
        noise = self.copy()
        noise.spec = S_opt*self.ar1_spec(self.freq, r_opt)
        
        return noise

    def getConfidenceInterval(self, confidenceValue):
        """
        Get confidence intervals.
        
        Confidence intervals: the ratio the estimated spectrum to the true spectrum is 
        distributed like $\chi^2_{\nu}/\nu$ where $\nu$ is the effective degrees of freedom.
        """
        from scipy.stats import chi2
        
        nu = self.edof
        
        conf_int = self.copy()
        conf_int.spec = self.spec*chi2.ppf((1+confidenceValue)/2, nu)/chi2.ppf(.5, nu)

        return conf_int

    def getConfidence(self, noise_model):
        """
        Get confidence value for each estimate.        
        """
        from scipy.stats import chi2

        freq = self.freq
        spec = self.spec
        dof  = self.edof

        # confidence of each estimate
        conf = 2*chi2.cdf(chi2.ppf(.5, dof)*self.spec/noise_model.spec,dof) - 1
        conf[conf < 0] = 0

        return conf


    def ftest(self):    # line testing
        '''
        Compute F-test for single spectral line components
        at the frequency bins given by the mtspec routines. 
    
        From German Prieto.
    
        Line test for periodic components. Assume that the lines have the form
            $$X(t) = B \mathrm{e}^{2\pi i f_0 t} + \eta(t),$$
        where $\eta$ represents white noise. The estimate for the line amplitude is
            $$\hat{B}(f_0) = \frac{\sum_{k=0}^{K-1} U_k(0) Y_k(f_0)}{\sum_{k = 0}^{K-1}U_k^2(0)},$$
        where $U_k(f)$ are the FTs of the DPSSs $w_k(t)$.

        The variance explained by the line is
            $$\theta = |\hat{B}(f_0)|^2 \sum_{k=0}^{K-1}U_k^2(0)$$
        and the remaining variance is
            $$\psi = \sum_{k=0}^{K-1}\left| Y_k(f_0) - \hat{B}(f_0)U_k(0)\right|^2.$$
        The test statistic 
            $$F(f) = (K-1)\frac{\theta}{\psi}$$
        obeys a Fisher-Snedecor law with 2 and $2(K-1)$ degrees of freedom.
    
        :return F: :class:`numpy.ndarray`
            vector of f-test values
        :return p: :class:`numpy.ndarray`
            vector of percentiles
        '''
        from scipy.stats import f as fdist
    
        nfft  = self.nfft
        kspec = self.number_of_tapers
        nf    = self.Nfreq
        yk    = self.eigenFT
    
        Uk0 = self.dpss.sum(axis=0)
    
        # mean amplitude of line components at each frequency
        B = np.sum(Uk0[np.newaxis,:]*yk[:nf,:],axis=1)/np.sum(Uk0**2)
    
        # F test
        #   numerator: model variance
        #   denominator: misfit
        fstat = (((kspec-1) * np.abs(B)**2 * sum(Uk0**2)) / 
            np.sum(np.abs(yk[:nf,:] - B[:,np.newaxis]*Uk0[np.newaxis,:])**2, axis=1))
    
        pval = fdist.cdf(fstat, 2, 2*(kspec-1))

        return pval, fstat

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

        if self.Hk is None: self.Hk = np.fft.rfft(h, nfft, 1)
        Hk = self.Hk

        Hk0 = np.real(Hk[:, 0][:, newaxis])
        Cl = (np.sum(Jk * Hk0, axis=0) / np.sum(Hk0 ** 2))[newaxis, :]

        F = (np.squeeze((K - 1) * np.abs(Cl) ** 2 * np.sum(Hk0 ** 2) 
                / np.sum(abs(Jk - Hk0 * Cl) ** 2, axis=0)))

        lines = np.nonzero(
            np.logical_and(
                S > dof / chi2.ppf((1 - chiTestCutoff) / 2, dof) * S_noiseModel,
                F > fdist.ppf(fTestCutoff, 2, 2 * K - 2)))[0]

        self.S_line = Cl
        self.Fstat = F
        self.lines = lines
        self.fLines = self.f[lines]

    def removeLines(self, linesToRemove):
        r"""
        Reshape eigenft's around specified spectral lines.
         
        Line removal and reshaping:

        The eigenFT of the line described above is
            $$Y_k(f) = B U_k(f - f_0).$$
        We remove this from the raw eigenFTs to get an estimate of the continuous spectrum.

        The associated power is
            $$\sum_{m=0}^{\hat{N}-1} |Y_k(f_m)|^2 = \hat{N} |B|^2,$$
        which is independent of $k$. This power is put in a single frequency bin given by $f_m = f_0$.   
        
        :param lines: list-like
            list of lines to remove
        :param wk: :class:`numpy.ndarray`
            the dpss
        :param yk: :class:`numpy.ndarra
            the eigenspectra
        :param mu: :class:`numpy.ndarray`
            The dpss concetrations
        :param nfft: int
            Length of FFT. If None, assumes nfft is even and defaults to 2*(# of frequencies - 1)
        :param xvar: float
            Variance of original time series (for rescaling spectrum). Default: 1

        :return spec_background: :class:`MTMSpectrum`
            background spectrum: original spectrum with the lines removed
        :return spec_lines: :class:`MTMSpectrum`
            the line spectrum
        :return line_power: :class:`MTMSpectrum`
            the power in each line
        """
        background = self.copy()
        lines      = self.copy()
        line_power = self.copy()
    
        background.eigenFT = background.eigenFT.copy()
        lines.eigenFT      = np.zeros_like(background.eigenFT)
        line_power.eigenFT = np.zeros_like(background.eigenFT)
            
        background.spec = background.spec.copy()
        lines.spec      = np.zeros_like(background.spec)
        line_power.spec = np.zeros_like(background.spec)
            
        npts  = self.N
        nfft  = self.nfft
        kspec = self.number_of_tapers
        nf    = self.Nfreq    
        fnyq  = self.fn
        df    = self.df
        wk    = self.dpss    
        
        if len(linesToRemove) == 0:
            return background, lines, line_power
    
        # calculate the mean amplitude of line components at line frequencies
        Uk0 = wk.sum(axis=0)
        B = np.sum(Uk0[np.newaxis,:]*self.eigenFT[linesToRemove,:],axis=1)/np.sum(Uk0**2)
    
        # Compute the Uks (FTs of wks) to reshape
        # The Uks are normalized to have unit integral
        Uk = np.fft.fft(wk, n=nfft, axis=0)
    
        # remove mean value for each spectral line
        j = np.arange(0, nfft)
       
        for i, line in enumerate(linesToRemove):
            background.eigenFT -= B[i]*Uk[j - line,:]
            lines.eigenFT      += B[i]*Uk[j - line,:]
        
#         from IPython.core.debugger import Tracer
#         Tracer()()
        background.adaptspec()
        lines.spec = (np.sum(background.weights**2*np.abs(lines.eigenFT[:nf])**2, axis=1) 
                        / np.sum(background.weights**2, axis=1)) 
    
        # get the line spectrum
        line_power.spec[linesToRemove] = nfft*np.abs(B)**2
    
        # double the power in positive frequencies
        background.spec[1:-1] *= 2
        line_power.spec[1:-1] *= 2
        lines.spec[1:-1] *= 2
        if np.mod(nfft,2) == 1: # if nfft is odd, double the power at the Nyquist frequency
            background.spec[-1] *= 2
            line_power.spec[-1] *= 2
            lines.spec[-1] *= 2

        # resscale to match original variance
        sscal = np.sum((background.spec + lines.spec)*df)
        background.spec *= self.xvar/sscal
        line_power.spec *= self.xvar/sscal
        lines.spec *= self.xvar/sscal

        return background, lines, line_power
        
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
        x = np.real(a * exp(2j * pi * f0 * t))

        if data is None:
            data = self.data

        rmse = np.sum((x - data) ** 2)

        return a, x, rmse, norm(H0 * ma.asmatrix(a).T), norm(H2 * ma.asmatrix(a).T), norm(H4 * ma.asmatrix(a).T)


def medianSmooth(x, width):
    '''
    Median smoothing of spectrum without zero padding.
    
    Edges are handled by shrinking the width of the smoother.
    
    :param x: :class:`numpy.ndarray`
        The spectrum to be smoothed
    :param width: integer
        Width of the filter (in bins)
    :param df: float
        frequency spacing for the spectrum. Defaults to df = 1

    :return y: :class:`numpy.ndarray`
        Smoothed spectrum
    '''

    N = len(x)

#     from IPython.core.debugger import Tracer
#     Tracer()()
    y = np.zeros(x.shape)
    for n in range(N):
        ix0 = max(n-width,0)
        ixf = min(n+width,N)

        y[n] = np.median(x[ix0:ixf])

    return y
