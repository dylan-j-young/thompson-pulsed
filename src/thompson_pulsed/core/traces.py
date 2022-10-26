# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:45:40 2022

@author: dylan
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import keyword

__all__ = ['Sequence', 'Time_Multitrace', 'MT_Phasor', 'MT_Phase', 
           'Frequency_Multitrace', 'Frequency_Sequence', 'SequenceLoadException']

class Sequence:
    """
    Stores information for a single sequence of an experiment. This is a generic
    class definition with variable-name attributes. 
    
    A Sequence object is required to have a time array, with the name ``t``. This
    array is detected and then used as the time record for all other arrays
    when packaged into Time Traces.
    
    Optionally, a Sequence object may have a trigger array, with the name ``trig``.
    This array is detected and then used to mark indices where a trigger is
    found.
    """
    def __init__(self, t, **kwargs):
        self.t = t
        for key in kwargs:
            vars(self)[key] = Time_Multitrace(t, kwargs[key])

        # Optional: find trigger array and record triggered indices
        self.has_triggers = ('trig' in kwargs)
        if self.has_triggers:
            self.triggers = self._mark_triggers()
        else:
            self.triggers = None
    
    @classmethod
    def load(cls, file, parser):
        """
        Generates a ``Sequence`` object by extracting data from a given file using
        the proper parser function. This is a class method so can be called
        directly on the class ``Sequence``.

        Parameters
        ----------
        file : str
            Path to the file in question.
        parser : function
            Parser used to extract relevant data from file.

        Returns
        -------
        An instance of ``Sequence`` in the form ``Sequence(t, **data_dict_from_file)``
        """
        data, dataset_names = parser(file)
        
        if data.size == 0:
            raise SequenceLoadException()
        
        # Find time array
        try:
            i_t = dataset_names.index('t')
        except ValueError:
            print('No time array with name "t" detected in file.')
            raise
        t = data[i_t]
        
        # Assign other arrays to variable-name attributes to kwargs
        kwargs = {}
        for i in range(len(dataset_names)):
            if i == i_t:
                continue
            dataset_name = dataset_names[i]
            if not dataset_name.isidentifier() or keyword.iskeyword(dataset_name) or dataset_name == 'None':
                raise SyntaxError(
                    f'"{dataset_name}" is not a valid variable name.'
                    )
            kwargs[dataset_name] = data[i]
        
        return( cls(t, **kwargs) )
    
    def _mark_triggers(self, slope=1):
        """
        Given a TTL Time Trace object stored in self.trig, detect all triggers
        within the trace and record their indices in a 1D np array.

        Parameters
        ----------
        slope : float, optional
            Specifies whether to look for a rising (1) or falling (-1) trigger.
            The default is 1.

        Returns
        -------
        triggers : 1D ndarray
        """
        trig_level = np.max(self.trig.V) / 2
        trig_min_spacing = 1e-6 # 1 us minimum distance between triggers
        
        triggers = np.where( (self.trig.V[1:] > trig_level) & \
                            (self.trig.V[:-1] < trig_level) )[0]
        if triggers.size > 1 and np.min(triggers[1:]-triggers[:-1]) < trig_min_spacing / self.trig.dt:
            raise Exception(
                'Two triggers closer than 1 us detected. Possible bad triggers.'
                )
        
        return(triggers)

class MT:
    """
    Generic multitrace object with a 1D x array and nD y array, where the LAST
    index corresponds to x.
    """
    def __init__(self, X, Y):
        """
        Initialized a MT object.

        Parameters
        ----------
        X : List of form [str, 1D np array]
        Y : List of form [str, nD np array]. Array looks like [..., x] with
            data points of arbitrary (d>=1) dimension. axis -1 corresponds to x.

        Returns
        -------
        None.
        """
        x_str, x = X
        y_str, y = Y

        # Check shape agreement and assign to object
        min_x_shape = min( x.shape[0], y.shape[-1] )

        # Assign data to custom attributes
        vars(self)[x_str] = x[:min_x_shape]
        vars(self)[y_str] = y[..., :min_x_shape]

        # Keep track of attribute names
        self.x_attr = x_str
        self.y_attr = y_str

    def chop(self, tol=10):
        """
        If there are small real or imaginary components of self.y, chop them
        off. For instance, 0.25 + 1e-14 * 1j gets transformed to 0.25.
        """
        y = vars(self)[self.y_attr]
        if np.max(np.imag(y)) < 10**-tol:
            y_new = np.round(np.real(y), 10)
        else:
            y_new = np.round(np.real(y), 10) + 1j * np.round(np.imag(y), 10)
        vars(self)[self.y_attr] = y_new
        return( self )
        

class Time_Multitrace(MT):
    """
    Stores information for multiple time traces. These multiple traces are
    assumed to be packed in a single numpy array of arbitrary dimension, where
    the LAST index corresponds to time.
    """
    def __init__(self, t, V, dV=None):
        """
        Initializes a Time_Multitrace object.

        Parameters
        ----------
        t : 1D np array
            time points for data.
        V : np array [..., t]
            data points of arbitrary (d>=1) dimension. axis -1 corresponds to time.
        dV : np array, shape = self.V.shape, optional
            uncertainties for self.V. If None, assume no uncertainties. Default
            is None.

        Returns
        -------
        None.
        """
        super().__init__( ['t',t], ['V',V] )

        # Check shape agreement and assign to object
        min_time_shape = min( t.shape[0], V.shape[-1] )
        if dV is not None:
            assert V.shape == dV.shape
        self.dV = dV[..., :min_time_shape] if (dV is not None) else None


    # t0, dt, T, dim are getter functions that read out values from self.t, self.V
    @property
    def t0(self):
        """
        Getter function that reads out initial time point. Call using ``MT.t0``.
        """
        return self.t[0]

    @property
    def dt(self):
        """
        Getter function that reads out the spacing between time point. Call using 
        ``MT.dt``.
        """
        return self.t[1]-self.t[0]

    @property
    def T(self):
        """
        Getter function that reads out the duration of the trace. Call using
        ``MT.T``.
        """
        return self.t[-1]-self.t[0] + self.dt

    @property
    def dim(self):
        """
        Getter function that reads out the dimensions of ``MT.V``. Call using
        ``MT.dim``.
        """
        return len(self.V.shape)

    ###
    # GENERICALLY USEFUL FUNCTIONS
    
    def set(self, V, dV = None):
        """
        Apply a new V (and dV) to a multitrace object, keeping the time values
        untouched.

        Parameters
        ----------
        V : ndarray
            New V array for trace
        dV : ndarray, optional
            New dV array for trace

        Returns
        -------
        Class of type self (base class Time_Multitrace)
        """

        if (type(V) != type(self.V)):
            raise( TypeError(f"V must have type {type(self.V)} to write to this object. It currently has type {type(V)}.") )
        if (V.shape[-1] != self.V.shape[-1]):
            raise( TypeError(f"V must have shape {self.V.shape[-1]} along the time axis to write to this object. It currently has shape {V.shape[-1]} along the time axis.") )

        if hasattr(self, 'dV'):
            if (type(dV) != type(None)) and (type(dV) != type(self.V)):
                raise( TypeError(f"dV must have type {type(None)} or type {type(self.V)} to write to this object. It currently has type {type(dV)}.") )
            if (dV is not None) and (dV.shape != V.shape):
                raise( TypeError(f"dV must have shape {V.shape} to write to this object. It currently has shape {dV.shape}.") )
            return( type(self)(self.t, V, dV=dV) )
        return( type(self)(self.t, V) )


    ###
    # FUNCTIONS WITH SPECIFIC USES

    def average_over(self, axis):
        """
        Given a time multitrace and a specific axis, perform a (potentially 
        weighted) average over that dimension and return a multitrace with one
        fewer dimension and statistics.

        Parameters
        ----------
        axis : int
            Axis to average over.

        Returns
        -------
        Class of type self (base class Time_Multitrace)
        """
        if self.dV is None:
            # No existing statistics. Generate unweighted average and uncertainty
            V_avg = np.average(self.V, axis=axis)
            dV_avg = np.std(self.V, axis=axis) / np.sqrt(self.V.shape[axis])
        else:
            # Generate weighted average and stdmean [using biased weighted estimator]
            weights = 1/self.dV**2
            V_avg = np.average(self.V, axis=axis, weights=weights)
            idx = (slice(None),) * axis + (None,) + (slice(None),) * (self.dim-axis-1)

            dV_avg = np.sqrt(np.average((self.V - V_avg[idx])**2, axis=axis, weights=weights))

        return( self.set(V_avg, dV=dV_avg) )

    def bin_trace(self, t_bin, t0=None):
        """
        Given a time multitrace and a time t_bin, bins the multiple time traces
        into subtraces with length t_bin in time. Given a k+1 dimensional
        multitrace of the form [d1, ..., dk, t], returns a k+2 dimensional
        multitrace of the form [d1, ..., dk, bin, t].
        
        Parameters
        ----------
        t_bin : float
            Binning time in units of self.t

        t0 : float, optional
            Specifies what time to start binning. If None, sets t0 = self.t[0].
            Default is None.

        Returns
        -------
        Class of type self (base class Time_Multitrace)
        """
        # Parameter checks
        if t0 is None or t0 < self.t[0]:
            t0 = self.t[0]
        elif t0 > self.t[-1]:
            raise ValueError("Error: t0 is out of bounds!")
        if t_bin < self.dt or t_bin > self.T:
            raise ValueError("Error: t_bin is out of bounds!")
        
        # Find bin starting index i0
        i0 = round( (t0-self.t[0])/self.dt )

        # Calculate bin parameters
        n_pts_raw = self.V.shape[-1] - i0
        n_bin_pts = round( t_bin/self.dt )
        n_bins = int( n_pts_raw / n_bin_pts )
        n_pts = n_bins * n_bin_pts
        
        # Reshape array
        V = self.V[..., i0:i0+n_pts]
        new_shape = V.shape[:-1] + (n_bins, n_bin_pts)
        V = np.reshape(V, new_shape)
        
        # Take only time values for first bin
        t = self.t[i0:i0+n_bin_pts]
        
        return( type(self)(t, V) )

    def binned_average(self, t_bin, use_weights=True):
        """
        Assuming an evenly-spaced array, time-bins an array and then averages
        points within that bin, returning a Time_Multitrace across the full
        time series with fewer, averaged points.
        
        Parameters
        ----------
        t_bin : float
            Time window for binning and averaging. Units the same as self.t
        use_weights: bool, optional
            Determines whether or not to weight data points when averaging.
            If True, attempts to define w = 1/dV**2 and otherwise uses equal
            weighting. If False, uses equal weighting. Default is True.
        
        Returns
        -------
        Time_Multitrace, representing a binned average of self.
        """
        bin_MT = self.bin_trace(t_bin)
        n_bins = bin_MT.V.shape[-2]
        n_binpts = bin_MT.V.shape[-1]
        t_bin_actual = n_binpts * self.dt
        
        t_averaged = self.t[0] + t_bin_actual * (1/2 + np.arange(n_bins))
        if (self.dV is not None) and (0 not in self.dV) and use_weights: 
            V_averaged = np.average(bin_MT.V, axis=-1, weights=1/bin_MT.dV**2)
            dV_averaged = 1/np.sqrt(np.sum(1/bin_MT.dV**2, axis=-1))
        else:
            V_averaged = np.average(bin_MT.V, axis=-1)
            dV_averaged = np.std(bin_MT.V, axis=-1)/np.sqrt(bin_MT.V.shape[-1])
        
        return( type(self)(t_averaged, V_averaged, dV=dV_averaged) )
    
    def collapse(self):
        """
        Returns a MT with the same data as self, but with all non-time
        dimensions collapsed into a single dimension. Reshaping is done with
        np.reshape(), which maintains dictionary order in the indices.
        
        Parameters
        ----------

        Returns
        -------
        Class of type self (base class Time_Multitrace)
        """
        new_shape = (np.prod(self.V.shape[:-1]), self.V.shape[-1])
        V_new = np.reshape(self.V, new_shape)
        dV_new = None if (self.dV is None) else (np.reshape(self.dV, new_shape))
        return( self.set(V_new, dV_new) )

    def fft(self, t_pad=None):
        """
        Returns a Frequency Multitrace corresponding to the given time
        multitrace. Optionally, allows zero padding up to t_pad, which allows
        one to interpolate the raw fft to a resolution of 1/t_pad. Fundamental
        Fourier broadening of the spectrum is still limited by the time length
        of the original array, as expected of time-frequency uncertainty.
        
        Parameters
        ----------
        t_pad : float, optional
            Pad time in units of self.t. If None, no padding is performed.
            Default is None.

        Returns
        -------
        Frequency_Multitrace
        """
        if t_pad == None:
            t_pad = self.T
        
        pad_pts = round(t_pad / self.dt)
        pad_tuple = ((0,0),) * (self.dim-1) + ((0, pad_pts-self.t.size),)
        V_pad = np.pad(self.V, pad_tuple )
        
        f_raw = np.fft.fftfreq(pad_pts, d=self.dt)
        Vf_raw = np.fft.fft(V_pad)
        
        f = np.fft.fftshift(f_raw, axes=-1)
        Vf = np.fft.fftshift(Vf_raw, axes=-1)

        return( Frequency_Multitrace(f, Vf) )
    
    def iq_demod(self, f_demod, filt='butter', order=4, f_cutoff=None, sign=1):
        """
        Performs an IQ demodulation on the multitrace with frequency f_demod.
        The scipy function filtfilt applies the given filter in the forward-
        time direction once, then applies it again in the backwards-time
        direction. This ensures the next applied filter has no net phase delay
        at the expense of preserving causality.

        Parameters
        ----------
        f_demod : float
            frequency to demodulate signal by. Units are period units in t
        filt : string, optional
            Describes which scipy filter to use. The default is 'butter'.
        order : int, optional
            Describes which order filter to use. The default is 4.
        f_cutoff : float, optional
            Describes the cutoff frequency of the filter. If None, function
            assumes f_cutoff to be f_demod. The default is None.

        Returns
        -------
        MT_Phasor(Time_Multitrace), representing demodulated phasors
        """
        if type(f_demod) == np.ndarray and f_demod.shape == self.V.shape[:-1]:
            # Different f_demod for each run of the Multitrace
            LO_x = np.cos(2*np.pi*f_demod[..., None]*self.t)
            LO_y = np.sin(2*np.pi*f_demod[..., None]*self.t)
        elif type(f_demod) == int or type(f_demod) == float:
            # Same f_demod for each run of the Multitrace
            LO_x = np.cos(2*np.pi*f_demod*self.t)
            LO_y = np.sin(2*np.pi*f_demod*self.t)
        else:
            raise TypeError("f_demod must be number or ndarray")
        
        # Generate 4th order Butterworth filter with cutoff at demod frequency
        if not f_cutoff:
            f_cutoff = f_demod if type(f_demod) != np.ndarray else f_demod[(0,)*(self.dim-1)]
        f_nyquist = 1/(2 * self.dt)
        wn = f_cutoff / f_nyquist
        n = order
        # TODO: give option to use different filter
        b,a = signal.butter(n, wn)
        
        V_x = signal.filtfilt(b,a, self.V*LO_x)
        V_y = signal.filtfilt(b,a, self.V*LO_y)

        return( MT_Phasor(self.t, V_x + sign * 1j * V_y) )
    
    def moving_average(self, t_avg, use_weights=True):
        """
        Assuming an evenly-spaced array, performs a moving average with window
        t_avg.

        Parameters
        ----------
        t_avg : float
            Time window for moving average. Units the same as self.t
        use_weights: bool, optional
            Determines whether or not to weight data points when averaging.
            If True, attempts to define w = 1/dV**2 and otherwise uses equal
            weighting. If False, uses equal weighting. Default is True.

        Returns
        -------
        Time_Multitrace, representing a moving average of self
        """
        # Determine number of points to average within t_avg
        k = int(t_avg/self.dt)
        if k < 1 or k > self.t.size:
            return( self )
        
        # Generate weights for average. If no errors exist, use equal weights
        w = (1/self.dV**2) if (self.dV is not None and use_weights) else (1 + 0*self.V)
        
        # Generate moving average
        V_sum = np.cumsum(self.V * w, dtype=float, axis=-1)
        V_sum[..., k:] = V_sum[..., k:] - V_sum[..., :-k]
        w_sum = np.cumsum(w, dtype=float, axis=-1)
        w_sum[..., k:] = w_sum[..., k:] - w_sum[..., :-k]
        shift = int((k-1)/2)
        
        # Normalize, then truncate the first k-1 points as they are not part of the average
        V_avg_trunc = V_sum[..., k-1:] / w_sum[..., k-1:]
        w_trunc = w_sum[..., k-1:]
        
        # Pad arrays with endpoint values to maintain shapes
        pad_width_V = ((0,0),) * (V_avg_trunc.ndim-1) + ((shift, k-shift-1),)
        V_avg = np.pad(V_avg_trunc, pad_width_V, mode='edge')
        dV_avg = np.pad(1/np.sqrt(w_trunc), pad_width_V, mode='edge')
        
        return( self.set(V_avg, dV=dV_avg) )
    
    def truncate(self, t_min=None, t_max=None):
        """
        Truncates the Time_Multitrace arrays to include only times t such that
        t_min <= t <= t_max.
        
        Parameters
        ----------
        t_min : float, optional
            Lower bound for truncation. If t_min is None, set t_min to t[0].
            Default is None.
        t_max : float, optional
            Upper bound for truncation. If t_max is None, set t_max to t[-1].
            Default is None.
        
        Returns
        -------
        Time_Multitrace, representing a time-truncated version of self.
        """
        if t_min is None:
            t_min = self.t[0]
        if t_max is None:
            t_max = self.t[-1]
            
        idx = np.where((self.t >= t_min) & (self.t <= t_max))[0]
            
        if self.dim > 1:
            V = self.V[...,idx]
            dV = None if self.dV is None else self.dV[...,idx]
        else:
            V = self.V[idx]
            dV = None if self.dV is None else self.dV[idx]
            
        return( type(self)(self.t[idx], V, dV=dV) )
        
        
class MT_Phasor(Time_Multitrace):
    """
    A multitrace object built off of ``Time_Multitrace`` which represents
    a time series of complex phasors, carrying with it phasor-specific
    class methods.
    """
    
    def phase(self, unwrap=True):
        """
        Given a MT_Phasor trace or phasors, generates an MT_Phase trace of
        phases.
        
        Parameters
        ----------
        unwrap : bool, optional
            Describes whether or not to unwrap the phases.

        Returns
        -------
        MT_Phase(Time_Multitrace) with same shape as self
        """
        return( MT_Phase(self.t, np.angle(self.V), unwrap=unwrap) )
    
    def mag(self):
        """Generates a Time_Multitrace of the phasor magnitudes.
        
        Parameters
        ----------

        Returns
        -------
        Time_Multitrace, with same shape as self
        """
        return( Time_Multitrace(self.t, np.abs(self.V), dV=self.dV) )

    def mag2(self):
        """Generates a Time_Multitrace of the phasor magnitudes squared.
        
        Parameters
        ----------

        Returns
        -------
        Time_Multitrace, with same shape as self
        """
        return( Time_Multitrace(self.t, np.abs(self.V)**2, 
            dV = (2*np.abs(self.V) * self.dV) if (self.dV is not None) else None ) )
    
    def real(self):
        """Generates a Time_Multitrace of the real components of the phasors.
        
        Parameters
        ----------

        Returns
        -------
        Time_Multitrace, with same shape as self
        """
        return( Time_Multitrace(self.t, np.real(self.V), dV=self.dV) )
    
    def imag(self):
        """Generates a Time_Multitrace of the imaginary components of the phasors.
        
        Parameters
        ----------

        Returns
        -------
        Time_Multitrace, with same shape as self
        """
        return( Time_Multitrace(self.t, np.imag(self.V), dV=self.dV) )

class MT_Phase(Time_Multitrace):
    """
    A multitrace object built off of ``Time_Multitrace`` which represents
    a time series of phase values, carrying with it phase-specific
    class methods.
    """

    def __init__(self, t, V, dV=None, unwrap=True):
        super().__init__(t,V,dV)
        
        if unwrap:
            self._unwrap()
            
    def _unwrap(self):
        """
        Unwraps the phase trace by detecting jumps in phase of larger than
        pi in a single time step, and correct.

        Returns
        -------
        None.
        """
        phase = self.V
        wraps = np.rint( (phase[..., 1:]-phase[..., :-1])/(2*np.pi) )
        change = np.concatenate((np.zeros( phase.shape[:-1] + (1,) ), 
                                 -np.cumsum(wraps, axis=-1)*2*np.pi), axis = -1)
        self.V += change            
        
    def frequency(self, t_bin=None, t0=None, wfunc=None):
        """
        Calculates a linear regression slope in O(n) time, using a closed-form
        formula for least-squares that assumes evenly sampled points in time. 
        
        Parameters
        ----------
        t_bin : float, optional
            Describes what size window to bin over. If None, assumes a single
            bin for the whole time trace. Default is None.
        t0 : float, optional
            Describes a start time to bin 
        wfunc : scalar function (np array compatible), optional
            Describes a weight map to apply to frequency measurement. Domain of
            wfunc assumed to vary from 0 to 1. Example: wfunc = lambda x:
            np.sin(np.pi*x)**2. If None, assumes a constant weight function.
            Default is None.

        Returns
        -------
        freqs : ndarray of floats
            Estimated frequencies of Phase traces, expressed in Hz (cycles/sec).
            Shape = self.V.shape[:-1]
        """
        if t_bin is not None:
            return( self.bin_trace(t_bin, t0=t0).frequency(t_bin=None, wfunc=wfunc) )
        elif wfunc is None:
            # No weights specified. Assume equal weighting
            y = self.V
            n = y.shape[-1]
            x = np.arange(1, n+1) - (n+1)/2
            dx = self.dt
            
            freqs = np.sum(x*y, axis=-1) / ( n*(n**2-1)/12 * dx ) * 1/(2*np.pi)
            return(freqs)
        else:
            # Calculate weights
            y = self.V
            n =  y.shape[-1]
            x = np.arange(n)/n + 1/(2*n) # x array from 0 to 1
            w = wfunc(x)
            dx = self.T
            
            xybar = np.average(x*y, axis=-1, weights=w)
            xbar = np.average(x, axis=-1, weights=w)
            ybar = np.average(y, axis=-1, weights=w)
            x2bar = np.average(x**2, axis=-1, weights=w)
            
            freqs = (xybar - xbar*ybar)/(x2bar - xbar**2) * 1/dx * 1/(2*np.pi)
            return(freqs)

class Frequency_Multitrace(MT):
    """
    Stores information for multiple frequency traces. These multiple traces are
    assumed to be packed in a single numpy array of arbitrary dimension, where
    the LAST index corresponds to frequency.
    """
    def __init__(self, f, V, dV=None):
        # Generate MT with attributes f, V
        super().__init__( ['f',f], ['V',V] )
        
        if dV is not None:
            assert V.shape == dV.shape
        self.dV = dV if (dV is not None) else None

    @property
    def f0(self):
        return self.f[0]

    @property
    def df(self):
        return self.f[1]-self.f[0]

    @property
    def F(self):
        return self.f[-1]-self.f[0] + self.df

    @property
    def dim(self):
        return len(self.V.shape)

    def mag2(self):
        V = np.abs(self.V)**2
        return( Time_Multitrace(self.f, np.abs(self.V)**2) )

    def ifft(self):
        """
        Returns a Time Multitrace corresponding to the given frequency
        multitrace.
        
        Parameters
        ----------

        Returns
        -------
        Time_Multitrace
        """
        n_pts = round( self.F / self.df )
        t = np.arange(0, n_pts) / self.F
        Vt = np.fft.ifft(np.fft.ifftshift(self.V, axes=-1))

        return( Time_Multitrace(t, Vt) )

class Frequency_Sequence:
    def __init__(self, f, **kwargs):
        self.f = f
        for key in kwargs:
            vars(self)[key] = Frequency_Multitrace(f, kwargs[key])
    
    @classmethod
    def load(cls, file, parser):
        """
        Generates a Frequency_Sequence class by extracting data from a give
        file using the proper parser function. This is a class method so can
        be called directly on the class Frequency_Sequence.

        Parameters
        ----------
        file : str
            Path to the file in question.
        parser : function
            Parser used to extract relevant data from file.

        Returns
        -------
        An instance of Frequency_Sequence in the form Frequency_Sequence(f, **data_dict_from_file)
        """
        data, dataset_names = parser(file)
        
        # Find time array
        try:
            i_f = dataset_names.index('f')
        except ValueError:
            print('No time array with name "f" detected in file.')
            raise
        f = data[i_f]
        
        # Assign other arrays to variable-name attributes to kwargs
        kwargs = {}
        for i in range(len(dataset_names)):
            if i == i_f:
                continue
            dataset_name = dataset_names[i]
            if not dataset_name.isidentifier() or keyword.iskeyword(dataset_name) or dataset_name == 'None':
                raise SyntaxError(
                    f'"{dataset_name}" is not a valid variable name.'
                    )
            kwargs[dataset_name] = data[i]
        
        return( cls(f, **kwargs) )
        
class SequenceLoadException(OSError):
    pass