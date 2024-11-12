"""
DefinedAEpTandZ0 (:mod:`skrf.media.definedAEpTandZ0`)
======================================================

Transmission line medium defined by A, Ep, Tand and Z0.

This medium is defined by attenuation A, relative permittivity Ep_r,
loss angle tand and characteristic impedance Z0.

Djirdjevic/Svennson dispersion model is provided for dielectric, default
behaviour is frequency invariant.

.. autosummary::
   :toctree: generated/

   DefinedAEpTandZ0
"""
from __future__ import annotations

import warnings

from numpy import imag, log, ones, pi, real, sqrt
from scipy.constants import c

from ..constants import NumberLike
from ..frequency import Frequency
from .media import Media


class DefinedAEpTandZ0(Media):
    r"""
    Transmission line medium defined by A, Ep, Tand and Z0.

    This medium is defined by attenuation `A`, relative permittivity `Ep_r`,
    loss angle `tand` and nominal impedance `Z0`.

    Djirdjevic [#Djordjevic]_ / Svennson [#Svensson]_ dispersion model
    is provided for dielectric. Default behaviour is frequency invariant.

    Parameters
    ----------
    frequency : :class:`~skrf.frequency.Frequency` object
        Frequency band of this transmission line medium
    z0_port : number, array-like, or None
        `z0_port` is the port impedance for networks generated by the media.
        If `z0_port` is not None, the networks generated by the media are
        renormalized (or in other words embedded) from the characteristic
        impedance z0 of the media to `z0_port`.
        Else if `z0_port` is None, the networks port impedances will be the raw
        characteristic impedance z0 of the media.
        (Default is None)
    A : number, array-like, default 0.0
        Attenuation due to conductor loss in dB/m/sqrt(Hz)
        The attenuation :math:`A(f)`at frequency :math:`f` is:

        .. math::

            A(f) = A\sqrt{\frac{f}{f_A}}

    f_A : number, default 1.0
        Frequency scaling in Hz for the attenuation. See A.
    ep_r : number, array-like, default 1.0
        Real part of the relative permittivity of the dielectric:
        :math:`\epsilon_r'=\Re[\epsilon]`.

        If `model='frequencyinvariant'`, the complex relative permittivity is:

        .. math::

            \epsilon_r(f) = \epsilon'_r + j \cdot \epsilon'_r \cdot \tan\delta

        if `model='djordjevicsvensson'`, the complex relative permittivity is:

        .. math::

            \epsilon_r(f) = \epsilon_\inf + m \cdot \ln{\frac{f_{high} + j f_\epsilon}{f_{low} + j f_\epsilon}}

        where :math:`\epsilon_\inf` is the permittivity value when frequency approaches infinity.
        In this case, the value of :math:`\epsilon_r` and :math:`\tan\delta` (tanD)
        are given for frequency :math:`f_\epsilon`:

        .. math::

            \epsilon_r(f_\epsilon) = \epsilon'_r+ j \cdot \epsilon'_r \cdot \tan\delta

    tanD : number, array-like, default 0.0
        Dielectric relative permittivity loss tangent :math:`\tan\delta`. See `ep_r`.
    z0 : number, array-like, default 50.0
        Quasi-static nominal impedance of the medium.
        Because of the distortion introduced by the conductor and dielectric
        losses, the characteristic impedance can be frequency-dependent with an
        imaginary part. The characteristic impedance is computed with an RLGC
        model.

        .. math::

            R = 2 Z_n \alpha_{conductor}

        .. math::

            L = \frac{Z_n \sqrt{\epsilon_r}}{C_0}

        .. math::

            G = \frac{2}{Z_n} \alpha_{dielectric}

        .. math::

            C = \frac{\sqrt{\epsilon_r}}{C_0 Z_n}

        .. math::

            Z_0 = \sqrt{\frac{R + j\omega L}{G + j\omega C}}

        where :math:`Z_n` is the nominal impedance and :math:`Z_0` is the
        characteristic impedance.
    Z0 : number, array-like, or None
        deprecated parameter, only emmit a deprecation warning.
    f_low : number, default 1e3, optional
        Low frequency in Hz for  for Djordjevic/Svennson dispersion model.
        See `ep_r`.
    f_high : number, default 1e12, optional
        High frequency in Hz for for Djordjevic/Svennson dispersion model.
        See `ep_r`.
    f_ep : number, default 1e9, , optional
        Specification frequency in Hz for  for Djirdjevic/Svennson dispersion model.
        `ep_r` and `tanD` parameters are specified for this frequency. See `ep_r`.
    model : string, 'frequencyinvariant' or 'djordjevicsvensson', optional
        Use Djirdjevic/Svennson wideband Debye dispersion model or not.
        Default is frequency invariant behaviour.
    \*args, \*\*kwargs : arguments and keyword arguments


    Examples
    --------
    >>> from skrf.media.definedAEpTandZ0 import DefinedAEpTandZ0
    >>> from skrf.frequency import Frequency
    >>> f = Frequency(75,110,101,'ghz')
    >>> DefinedAEpTandZ0(frequency=f, A=1, f_A=1e9, ep_r=3, z0=50)
    >>> DefinedAEpTandZ0(frequency=f, A=1, f_A=1e9, ep_r=3, tand=0.02, z0=50)
    >>> DefinedAEpTandZ0(frequency=f, A=1, f_A=1e9, ep_r=3, tand=0.02, z0=50,
                        f_low=1e3, f_high=1e12, f_Ep=1e9,
                        model='djordjevicsvensson')

    References
    ----------
    .. [#Svensson] C. Svensson and G. E. Dermer, "Time Domain Modeling of Lossy Interconnects,"
        IEEE Trans. Advanced Packaging, Vol. 24, No. 2, May 2001.
        https://doi.org/10.1109/6040.928754
    .. [#Djordjevic] A. R. Djordjevic, R. M. Biljic, V. D. Likar-Smiljanic, and T. K. Sarkar,
        "Wideband Frequency-Domain Characterization of FR-4 and Time-Domain Causality,"
        IEEE Trans. Electromagnetic Compatibility, Vol. 43, No. 4, November 2001.
        https://doi.org/10.1109/15.974647

    """

    def __init__(self, frequency: Frequency| None = None,
                 A: float = 0.0, f_A: float = 1.0,
                 ep_r: NumberLike = 1.0, tanD: NumberLike = 0.0,
                 z0_port: NumberLike | None = None,
                 z0: float = 50.0,
                 Z0: NumberLike | None = None,
                 f_low: float = 1.0e3, f_high: float = 1.0e12, f_ep: float = 1.0e9,
                 model: str = 'frequencyinvariant', *args, **kwargs):

        Media.__init__(self, frequency=frequency, z0_port=z0_port)
        self.A, self.f_A = A, f_A
        self.ep_r, self.tanD = ep_r, tanD
        self.Zn = z0
        self.f_low, self.f_high, self.f_ep = f_low, f_high, f_ep
        self.model = model

        # z0 is the nominal impedance. Compute characteristic impedance with
        # an RLGC model based on alpha conductor and alpha dielectric.
        self.R = 2 * self.Zn * self.alpha_conductor
        self.L = self.Zn * sqrt(self.ep_r) / c
        self.G = 2 / self.Zn * self.alpha_dielectric
        self.C = sqrt(ep_r) / c / self.Zn
        self.z0_characteristic = sqrt(
            (self.R + 1j * self.frequency.w * self.L) /
            (self.G + 1j * self.frequency.w * self.C))

        if Z0 is not None:
            # warns of deprecation
            warnings.warn(
                'Use of `Z0` in DefinedAEpTandZ0 initialization is deprecated.\n'
                '`Z0` has no effect. Use `z0` instead\n'
                '`Z0` will be removed in version 1.0',
              DeprecationWarning, stacklevel = 2)

    def __str__(self):
        f = self.frequency
        output = 'DefinedAEpTandZ0 medium.  %i-%i %s.  %i points'%\
            (f.f_scaled[0], f.f_scaled[-1], f.unit, f.npoints)
        return output

    def __repr__(self):
        return self.__str__()

    @property
    def ep_r_f(self) -> NumberLike:
        """
        Frequency dependent complex relative permittivity of dielectric.
        """
        ep_r, tand  = self.ep_r, self.tanD
        f_low, f_high, f_ep = self.f_low, self.f_high, self.f_ep
        f = self.frequency.f
        if self.model == 'djordjevicsvensson':
           # compute the slope for a log frequency scale, tanD dependent.
           k = log((f_high + 1j * f_ep) / (f_low + 1j * f_ep))
           fd = log((f_high + 1j * f) / (f_low + 1j * f))
           ep_d = -tand * ep_r  / imag(k)
           # value for frequency above f_high
           ep_inf = ep_r * (1. + tand * real(k) / imag(k))
           # compute complex permitivity
           return ep_inf + ep_d * fd
        elif self.model == 'frequencyinvariant':
            return ones(self.frequency.f.shape) * (ep_r - 1j*ep_r*tand)
        else:
            raise ValueError('Unknown dielectric dispersion model')

    @property
    def tand_f(self) -> NumberLike:
        """
        Frequency dependent dielectric loss factor.
        """
        ep_r = self.ep_r_f
        return -imag(ep_r) / real(ep_r)

    @property
    def alpha_conductor(self) -> NumberLike:
        """
        Losses due to conductor resistivity.
        """
        A, f_A, f = self.A, self.f_A, self.frequency.f
        return A * log(10)/20 * sqrt(f/f_A)

    @property
    def alpha_dielectric(self) -> NumberLike:
        """
        Losses due to dielectric
        """
        ep_r, tand = real(self.ep_r_f), self.tand_f
        f = self.frequency.f
        return pi*sqrt(ep_r)*f/c * tand

    @property
    def beta_phase(self) -> NumberLike:
        """
        Phase parameter
        """
        ep_r, f = real(self.ep_r_f), self.frequency.f
        return 2*pi*f*sqrt(ep_r)/c

    @property
    def gamma(self) -> NumberLike:
        """
        Propagation constant.

        See Also
        --------
        alpha_conductor : calculates conductor losses
        alpha_dielectric: calculates dielectric losses
        beta_phase      : calculates phase parameter
        """
        beta  = self.beta_phase
        alpha = self.alpha_conductor + self.alpha_dielectric
        return alpha + 1j*beta

    @property
    def z0_characteristic(self) -> NumberLike:
        """
        Characteristic Impedance, :math:`z_0`

        Returns
        -------
        z0_characteristic : np.ndarray
            Characteristic Impedance in units of ohms
        """
        return self._z0

    @z0_characteristic.setter
    def z0_characteristic(self, val):
        self._z0 = val
