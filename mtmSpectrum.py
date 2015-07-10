import numpy as np

__author__ = 'cwolfe'


class MTMSpectrum:
    """Class implementing Thompson's multi-taper method with line detection and reshaping"""

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
            self.nfft = int(2 ** (np.ceil(np.log2(self.N))))  # next power of two
            if self.nfft < 1.5 * self.N: self.nfft *= 2
        else:
            self.nfft = nfft

        # finished with setup, calculate the spectrum
        self.calcSpectrum(adaptTol, adaptMaxIts)

    def calcSpectrum(self, tol=None, MaxIts=20):
        from nitime.algorithms.spectral import dpss_windows

        self.dpss, lam = dpss_windows(self.N, self.nw, self.K)
        self.lk = lam[:, np.newaxis]

        self.f = np.fft.rfftfreq(self.nfft, d=self.dt)
        self.Nfreq = self.f.size
        self.df = self.f[1] - self.f[0]
        self.fn = self.f[-1]
        self.fr = self.fs / self.N

        self.Jk = np.fft.rfft(self.dpss * self.data, self.nfft, 1)
        self.Sk = np.abs(self.Jk) ** 2

        self.adaptWeights(tol, MaxIts)
        self.S_noise = self.S.copy()

    def adaptWeights(self, tol=None, MaxIts=20):
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

            if np.abs(S - S1).mean() < tol: break

            Stemp = S1  # swap S and S1
            S1 = S
            S = Stemp
        else:
            warnings.warn('Adaptive weight iteration did not converge')

        # effective degrees of freedom
        if self.useEffectiveDOF:
            self.dof = 2 * np.sum(b ** 2 * lk, axis=0) ** 2 / np.sum(b ** 4 * lk ** 2, axis=0)
        else:
            self.dof = np.tile(2 * self.K, (self.Nfreq))

        self.S = S
        self.bk = b

    def ar1(self, f, sig2, r):
        return sig2 * (1 - r ** 2) / (1 + r ** 2 - 2 * r * np.cos(np.pi * f))

    def lorentzian(self, f, sig2, gamma):
        return (2. / (np.pi * gamma)) * sig2 / (1 + f ** 2 / gamma ** 2)

    def getConfidence(self, confidenceValue, smoothingWidth, fitType='ar1', fitRange=None):
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
            idx = np.logical_and(f >= fitRange[0], f <= fitRange[1])

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

        Hk0 = np.real(Hk[:, 0][:, np.newaxis])
        Cl = (np.sum(Jk * Hk0, axis=0) / np.sum(Hk0 ** 2))[np.newaxis, :]

        F = np.squeeze((K - 1) * np.abs(Cl) ** 2 * np.sum(Hk0 ** 2) / np.sum(np.abs(Jk - Hk0 * Cl) ** 2, axis=0))

        lines = np.nonzero(
            np.logical_and(
                S > dof / chi2.ppf((1 - chiTestCutoff) / 2, dof) * S_noiseModel,
                F > fdist.ppf(fTestCutoff, 2, 2 * K - 2)))[0]

        self.S_line = Cl
        self.Fstat = F
        self.lines = lines
        self.fLines = self.f[lines]

    def removeLines(self, linesToRemove, lineWidth=None):
        #    import ipdb; ipdb.set_trace()
        S = self.S
        Hk = self.Hk
        Jk = self.Jk
        nw = self.nw
        Cl = self.S_line

        lines = np.asarray(linesToRemove)

        if lineWidth is None:
            width = np.tile(int(np.ceil(nw)), lines.shape)
        else:
            width = np.asarray(lineWidth, dtype=np.int) - 1

        Shat = self.S_noise

        for l0, W in zip(lines, width):
            l = range(max(l0 - W, 0), min(l0 + W + 1, S.size))
            Hloc = Hk[:, np.abs(l - l0)]
            Hloc[:, l < l0] = np.conj(Hloc[:, l < l0])
            Shat[l] = np.mean(np.abs(Jk[:, l] - Cl[0, l0] * Hloc) ** 2, axis=0)

        self.removedLines = lines
        self.fRemovedLines = self.f[lines]

    def resetLines(self):
        self.removedLines = None
        self.fRemovedLines = None
        self.S_noise = self.S.copy()




def medianSmooth(x,width):
    w = 2*np.floor(1+width/2)-1  # ensure width is an odd integer
    nw = (w-1)/2

    N = len(x)

    y = np.zeros(x.shape)
    for n in range(N):
        ix0 = max(n-nw,1)
        ixf = min(n+nw,N)

        y[n] = np.median(x[ix0:ixf])

    return y
