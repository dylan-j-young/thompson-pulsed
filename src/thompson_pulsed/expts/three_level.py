# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 18:00:35 2022

@author: dylan
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

from ..core import traces
from ..core.parsers import ni_pci5105
"""
Define experiment-specific functions
"""
def sinc(x):
    return(np.sinc(x/np.pi))
def sinc_symm(f, f0, A, T):
    return( A*(sinc(np.pi*(f-f0)*T)**2 + sinc(np.pi*(f+f0)*T)**2) )
def sinc_symm_fitter(T):
    """
    Returns a symmetric sinc function, with the sinc width fixed to T.
    """
    return( lambda f,f0,A: sinc_symm(f,f0,A,T) )
def cos(t, A, f, phi):
    return( A*np.cos(2*np.pi*f*t + phi) )
def moving_average(V, k):
    V_sum = np.cumsum(V, dtype=float, axis=-1)
    V_sum[..., k:] = V_sum[..., k:] - V_sum[..., :-k]
    shift = int((k-1)/2)
    V_avg_trunc = V_sum[..., k-1:]/k
    pad_width = ((0,0),) * (V_avg_trunc.ndim-1) + ((shift, k-shift-1),)
    return( np.pad(V_avg_trunc, pad_width) )
def notch(V, f_notch, Q, fs):
    b, a = signal.iirnotch(f_notch, Q, fs)
    return( signal.filtfilt(b,a,V,axis=-1) )

"""
Define object classes for experiment
"""
class Experiment:
    """
    Root object of a particular instance of the experiment.
    """
    def __init__(self, params):
        self.sequences = []
        self.data = None
        
        if params._all_params_defined():
            self.params = params
        else:
            raise Exception("Error: not all parameters defined!")

    def load_sequence(self, file):
        """
        Loads a text file from a single sequence into the experiment.
        """
        tries = 6
        delay = 0.1
        for i in range(tries):
            try:
                seq = traces.Sequence.load(file, ni_pci5105)
            except traces.SequenceLoadException as err:
                print(f'Try {i+1} to load {file} failed.')
                if i+1 == tries:
                    raise err
                time.sleep(delay)
                delay *= 2
                continue
            break
        
        self.sequences.append(seq)
        
    def preprocess(self, n_seqs = None, load = 'newest', premeasure = 0,
                   premeasure_interleaved = False, postmeasure = 0):
        """
        Given loaded sequences, preprocess data contained inside and store in
        expt.data.

        Parameters
        ----------
        n_seqs : int, optional
            Number of sequences to load. If None, loads all sequences
            available. The default is None.
        load : str, optional
            Specifies which sequences to load to the Data class. If 'oldest',
            loads the first n_seqs sequences. If 'newest', loads the last
            n_seqs sequences. The default is 'newest'.
        premeasure : int, optional
            Specifies number of premeasurement runs performed in sequence. 
            Default is 0.
        premeasure_interleaved : bool, optional
            Specifies whether or not premeasurement runs are interleaved with
            actual runs. If True, this overrides the premeasure count from
            the above optional argument and instead assumes one premeasure
            run for each actual run. Default is False.
        postmeasure : int, optional
            Specifies number of postmeasurement runs performed in sequence. 
            Default is 0.

        Returns
        -------
        None.
        """
        if not self.params:
            raise Exception("Error: need to set experiment parameters!")
        if not self.sequences:
            raise Exception("Error: no sequences loaded to experiment!")
        
        # Process optional args
        if not n_seqs or n_seqs > len(self.sequences):
            n_seqs = len(self.sequences)
            
        if load=='oldest':
            sequences = self.sequences[:n_seqs]
        elif load=='newest':
            sequences = self.sequences[-n_seqs:]
        else:
            sequences = self.sequences[-n_seqs:]

        # OPTIONAL ARG: override premeasure argument if interleaved specified
        if premeasure_interleaved:
            premeasure = 0
        
        # Define a single time array for all runs of the experiment
        self.params.dt = self.sequences[0].t[1] - self.sequences[0].t[0]
        i_run = round(self.params.t_run / self.params.dt)
        t = self.params.dt * np.arange(i_run)

        # Iterate over all sequences
        atom_runs, cav_runs = np.array([]), np.array([])
        preseq_cav, postseq_cav = None, None
        for i in range(len(sequences)):
            seq = sequences[i]
            
            # Extract runs from sequence
            if not seq.has_triggers:
                # Assume a single run for each sequence and extract points
                seq_atom_runs = np.array([seq.atom.V[..., :i_run]])
                seq_cav_runs = np.array([seq.cav.V[..., :i_run]])
            else:
                # Edge case: redefine i_run if dataset ends abruptly
                i_run = min(i_run, seq.atom.V.shape[-1] - seq.triggers[-1])
                
                # Extract runs from the single sequence using triggers as markers
                seq_atom_runs = np.array(
                    [seq.atom.V[..., trig:trig+i_run] for trig in seq.triggers]
                    )
                seq_cav_runs = np.array(
                    [seq.cav.V[..., trig:trig+i_run] for trig in seq.triggers]
                    )
                
            # Check for premeasure. preseq_cav.shape = (seq, run, t)
            if premeasure > 0 and seq_cav_runs.shape[0] > premeasure:
                preseq_cav = \
                    np.concatenate((preseq_cav, [seq_cav_runs[:premeasure]]), axis=0) \
                    if (preseq_cav is not None) \
                    else np.array([seq_cav_runs[:premeasure]])
                seq_cav_runs = seq_cav_runs[premeasure:]
                seq_atom_runs = seq_atom_runs[premeasure:]
            
            # Check for postmeasure. postseq_cav.shape = (seq, run, t)
            if postmeasure > 0 and seq_cav_runs.shape[0] > postmeasure:
                postseq_cav = \
                    np.concatenate((postseq_cav, [seq_cav_runs[-postmeasure:]]), axis=0) \
                    if (postseq_cav is not None) \
                    else np.array([seq_cav_runs[-postmeasure:]])
                seq_cav_runs = seq_cav_runs[:-postmeasure]
                seq_atom_runs = seq_atom_runs[:-postmeasure]
                
            # Check for interleaved premeasure. preseq_cav.shape = (seq, run, t)
            if premeasure_interleaved:
                preseq_cav = \
                    np.concatenate((preseq_cav, [seq_cav_runs[::2]]), axis=0) \
                    if (preseq_cav is not None) \
                    else np.array([seq_cav_runs[::2]])
                seq_cav_runs = seq_cav_runs[1::2]
                seq_atom_runs = seq_atom_runs[1::2]
                
            # Add remaining runs from sequence to the full arrays. (seq, run, t)
            if atom_runs.size > 0:
                atom_runs = np.concatenate((atom_runs, [seq_atom_runs]), axis=0)
                cav_runs = np.concatenate((cav_runs, [seq_cav_runs]), axis=0)
            else:
                atom_runs = np.array([seq_atom_runs])
                cav_runs = np.array([seq_cav_runs])
               
        # Process postmeasure. fb.shape = (seq,)
        fb = None
        if postmeasure > 0:
            fb_runs = Data(t, postseq_cav, None, self.params) \
                .track_cav_frequency_iq(collapse=False)
            if fb_runs.V.shape[1] > 1:
                # Can calculate statistics for dV
                fb_seqs = traces.Time_Multitrace(
                    fb_runs.t, np.mean(fb_runs.V, axis=1),
                    dV = np.std(fb_runs.V, axis=1)/np.sqrt(fb_runs.V.shape[1])
                )
                fb = np.average(fb_seqs.V, axis=-1, weights=1/fb_seqs.dV**2)
            else:
                # Don't attempt to calculate dV
                fb_seqs = traces.Time_Multitrace(
                    fb_runs.t, np.mean(fb_runs.V, axis=1)
                )
                fb = np.average(fb_seqs.V, axis=-1)
        
        # Process premeasure. fi.shape = (seq,)
        fi = None
        if premeasure > 0 or premeasure_interleaved:
            fi_runs = Data(t, preseq_cav, None, self.params, fb=fb) \
                .track_cav_frequency_iq(collapse=False)
            if fi_runs.V.shape[1] > 1:
                fi_seqs = traces.Time_Multitrace(
                    fi_runs.t, np.mean(fi_runs.V, axis=1),
                    dV = np.std(fi_runs.V, axis=1)/np.sqrt(fi_runs.V.shape[1])
                )
                fi = np.average(fi_seqs.V, axis=-1, weights=1/fi_seqs.dV**2)
            else:
                # Don't attempt to calculate dV
                fi_seqs = traces.Time_Multitrace(
                    fi_runs.t, np.mean(fi_runs.V, axis=1)
                )
                fi = np.average(fi_seqs.V, axis=-1)
        
        # Assign to data object
        self.data = Data(t, cav_runs, atom_runs, self.params, fi=fi, fb=fb)
            
class Parameters:
    def __init__(self):
        self.t_run = None
        self.t_bin = None
        self.t_drive = None
        # self.t_fft_pad = None
        self.t_cav_pulse = None
        self.f0_cav = None
        self.f0_atom = None
        # self.fft_fit = None
        self.demod_smoother = None
        
    def _all_params_defined(self):
        for attr in vars(self):
            if vars(self)[attr] is None:
                return(False)
        return(True)

class Data:
    """
    Stores preprocessed data. Accessed via the parent Experiment object. 
    """
    def __init__(self, t, cav_runs, atom_runs, params, fi=None, fb=None):
        self.t = t
        self.cav_runs = traces.Time_Multitrace(t, cav_runs) \
            if (cav_runs is not None) else None
        self.atom_runs = traces.Time_Multitrace(t, atom_runs) \
            if (atom_runs is not None) else None
        self.params = params
        
        # Initial and bare cavity frequency estimators. Shape: (seq,)
        self.fi = fi
        self.fb = fb
    
    def track_cav_frequency_iq(self, f_demod = None, out_fmt = 'MT', align = True, collapse = True, \
                               ignore_pulse_bins = True, moving_average = 0):
        """
        IQ demodulates cavity time traces, bins them, and fits their phase(t)
        with a linear regression to estimate instantaneous frequency. Multiple
        shots give statistics on these bins.
        
        Parameters
        ----------
        f_demod : float, optional
            Specifies what frequency to demodulate at. If None, f_demod is auto-set
            to self.params.f0_cav. The default is None.
        out_fmt : str, optional
            Specifies what format the data is outputted in. All methods other than
            the default are deprecated.
            'trace' : returns t, V as two separate np arrays
            'MT' : returns t, V as a single tp.Time_Multitrace (see below)
            Default is 'MT'
        align : boolean, optional
            Specifies whether or not to align the time trace bins to a potential
            pulsed cavity probe. If True, looks for the time 0 <= t < t_bin
            corresponding to the maximum value of the first cavity trace and
            truncates the trace to start at this time, then bins. If False,
            performs no truncation. Default is True.
        collapse : boolean, optional
            Specifies whether to collapse all degrees of freedom in the array
            except for the final two: [run,bin]. Default is True.
        ignore_pulse_bins : boolean, optional
            Specifies whether to ignore time bins that occur during a pulse.
            Assumes an integer number of t_bins in a t_cav_pulse. Default is
            True.
        moving_average: float, optional
            Specifies what timespan of moving average should be applied to the
            demodulated phasor, if any. If 0, does not perform an average.
            Default is 0.

        Returns
        -------
        cav_freqs : tp.Time_Multitrace
            .t : 1D np array [bin]
            .V : 2D np array [run,bin]
        """
        if self.cav_runs is None:
            raise Exception('cav_runs was not set!')
        
        # OPTIONAL ARG: Pull f_demod if not specified
        if f_demod is None:
            f_demod = self.params.f0_cav
            
        cav_runs = self.cav_runs
        cav_runs.V -= np.mean(cav_runs.V, axis=-1, keepdims=True)
        cav_phase = cav_runs.iq_demod(f_demod).phase()
        phase = cav_phase.V
        
        t_pulse = self.params.t_cav_pulse
        n_pulse_pts = round( t_pulse/cav_phase.dt )
        n_pts = phase.shape[-1]
        n_runs = np.prod(phase.shape[:-1])
        
        if int(self.params.t_cav_pulse/self.params.t_bin) > 1:
            # OPTIONAL ARG: Align to pulses
            n_bin_pts = round( self.params.t_bin/cav_runs.dt )
            
            use_phase_jumps = True
            if align and use_phase_jumps:
                # New method for aligning to pulses
                
                # Collapse into trace of t_pulse
                phase_diff = np.concatenate((np.zeros( phase.shape[:-1] + (1,) ),
                                         phase[...,1:]-phase[...,:-1]), axis = -1)
                n_pulses_temp = int(n_pts/n_pulse_pts)
                pulse_finder = np.reshape(
                    phase_diff[..., :n_pulses_temp*n_pulse_pts]**2,
                    (n_runs*n_pulses_temp, n_pulse_pts)
                    ).mean(axis=-2)
                
                # Find index with max alignment with phase jumps
                pulse_max = np.max(pulse_finder, axis=-1)
                tol = 0.9
                imin = np.argmax(pulse_finder > tol*pulse_max, axis=-1)
                imax = n_pulse_pts-1 - np.argmax(pulse_finder[...,::-1] > tol*pulse_max, axis=-1)
                i0_pulse = np.round( (imin+imax)/2 ).astype(int)
                i0 = i0_pulse + int(n_bin_pts/2) # Contain most of jump within a single bin
                
                cav_runs = traces.Time_Multitrace(cav_runs.t[i0:], cav_runs.V[...,i0:])
            elif align:                
                # Get first trace and find max (corresponding to one of the pulses)
                idx = (0,) * (cav_runs.dim-1) + (slice(None),)
                i_pulse = np.argmax(cav_runs.V[idx])
                
                # Find offset from t=0 to bin aligned to i0
                i0 = i_pulse % n_pulse_pts
                
                # Truncate the first part of the trace to align to pulses
                cav_runs = traces.Time_Multitrace(cav_runs.t[i0:], cav_runs.V[...,i0:])
                # print(f'i0 = {i0}. n_pulse_pts = {n_pulse_pts}. dt = {cav_runs.dt}.')
            
            
            # Bin cavity run traces, subtract mean from each bin, and demodulate
            cav_runs.V -= np.mean(cav_runs.V, axis=-1, keepdims=True)
            cav_phase = cav_runs.iq_demod(f_demod).phase()
            # if moving_average > 0:
            #     cav_phase = cav_phase.moving_average(moving_average)
            cav_bins = cav_phase.bin_trace(self.params.t_bin)
                                    
            # Get bin times
            bin_times = cav_runs.t[0] + self.params.t_bin * (0.5 + np.arange(cav_bins.V.shape[-2]))
            
            # Estimate cavity frequency in bins using linear regression
            cav_freq_vals = cav_bins.frequency()
            
            # OPTIONAL ARG: Remove pulse bins
            if ignore_pulse_bins:                
                # Get first trace and find max (corresponding to one of the pulses)
                idx = (0,) * (cav_runs.dim-1) + (slice(None),)
                i_pulse = np.argmax(cav_runs.V[idx])
                
                # Will ignore every [n_pulse_bins]-th bin
                n_pulse_pts = round( self.params.t_cav_pulse/cav_runs.dt )
                n_pulse_bins = round( n_pulse_pts / n_bin_pts )
                
                # Delete offending bins
                pulse_slice = slice(n_pulse_bins-1, None, n_pulse_bins)
                cav_freq_vals = np.delete(cav_freq_vals, pulse_slice, axis=-1)
                bin_times = np.delete(bin_times, pulse_slice, axis=-1)
        else:
            # Calculate frequency in bins of each pulse
            # Get MT_Phase trace for demodulated cavity data            
            runs_aligned = True
            single_index_align = False
            if runs_aligned:
                # Assume runs are aligned to each other
                # Find where pulses are
                if single_index_align == True:
                    i_pulse = np.unravel_index(
                        np.argmax(phase[...,1:]-phase[...,:-1]), phase[...,:-1].shape
                        )[-1]
                    
                    # Find bin points which align with cavity pulses
                    ip0 = np.round( i_pulse % n_pulse_pts ).astype(int)
                else:
                    # Collapse into trace of t_pulse
                    phase_diff = np.concatenate((np.zeros( phase.shape[:-1] + (1,) ),
                                             phase[...,1:]-phase[...,:-1]), axis = -1)
                    n_pulses_temp = int(n_pts/n_pulse_pts)
                    pulse_finder = np.reshape(
                        phase_diff[..., :n_pulses_temp*n_pulse_pts]**2,
                        (n_runs*n_pulses_temp, n_pulse_pts)
                        ).mean(axis=-2)
                    
                    # Find index with max alignment with phase jumps
                    pulse_max = np.max(pulse_finder, axis=-1)
                    tol = 0.9
                    imin = np.argmax(pulse_finder > tol*pulse_max, axis=-1)
                    imax = n_pulse_pts-1 - np.argmax(pulse_finder[...,::-1] > tol*pulse_max, axis=-1)
                    ip0 = np.round( (imin+imax)/2 ).astype(int)

                n_pulses = int( (n_pts - 1 - ip0) / n_pulse_pts ) 
    
                # Flatten arrays to be in the form [runs,t]
                # Then generate array of selected phase values at bin points
                n_pts_new = n_pulses*n_pulse_pts
                phase_flat_aligned_vals = np.reshape(
                    phase[..., ip0:ip0+n_pts_new], (n_runs,n_pts_new)
                    )
            else:
                # Assume runs are not aligned to each other
                # Find where pulses are
                if single_index_align == True:
                    i_pulse = np.argmax(phase[...,1:]-phase[...,:-1], axis=-1)  
                    
                    # Find bin points which align with cavity pulses
                    ip0 = np.round( i_pulse % n_pulse_pts ).astype(int)
                else:
                    # Collapse into trace of t_pulse
                    phase_diff = np.concatenate((np.zeros( phase.shape[:-1] + (1,) ),
                                             phase[...,1:]-phase[...,:-1]), axis = -1)
                    n_pulses_temp = int(n_pts/n_pulse_pts)
                    pulse_finder = np.reshape(
                        phase_diff[..., :n_pulses_temp*n_pulse_pts]**2,
                        (n_runs*n_pulses_temp, n_pulse_pts)
                        ).mean(axis=-2)
                    
                    # Find index with max alignment with phase jumps
                    pulse_max = np.max(pulse_finder, axis=-1)
                    tol = 0.9
                    imin = np.argmax(pulse_finder > tol*pulse_max, axis=-1)
                    imax = n_pulse_pts-1 - np.argmax(pulse_finder[...,::-1] > tol*pulse_max, axis=-1)
                    ip0 = np.round( (imin+imax)/2 ).astype(int)

                n_pulses = int( (n_pts - 1 - np.amax(ip0)) / n_pulse_pts ) 
    
                # Flatten arrays to be in the form [runs,t]
                # Then generate array of selected phase values at bin points
                ip0_flat = np.reshape(ip0, n_runs)
                phase_flat = np.reshape(phase, (n_runs, n_pts))
                phase_flat_aligned_vals = np.zeros((n_runs, n_pulses*n_pulse_pts))
                for run in range(n_runs):
                    i0 = ip0_flat[run]
                    phase_flat_aligned_vals[run] = phase_flat[run, i0:i0+n_pulses*n_pulse_pts]

            # Construct phase object with aligned pulses
            t_aligned = cav_phase.t[:n_pulses*n_pulse_pts]
            phase_flat_aligned = traces.MT_Phase(t_aligned, phase_flat_aligned_vals)

            wfunc = lambda x: np.sin(np.pi * np.cos(np.pi/2*(1-x)) )**2
            # wfunc = None
            cav_freq_vals_flat = phase_flat_aligned.frequency(t_bin=t_pulse, wfunc=wfunc)
            
            n_bin_pulses = int(self.params.t_bin/t_pulse)
            n_bins = int(n_pulses/n_bin_pulses)
            cav_freq_vals_bins = cav_freq_vals_flat[:,:n_bins*n_bin_pulses] \
                .reshape((n_runs, n_bins, n_bin_pulses)) \
                .mean(axis=-1)            
            cav_freq_vals = cav_freq_vals_bins \
                .reshape(phase.shape[:-1] + (n_bins,))
            bin_times = cav_phase.t[0] + self.params.t_bin * (0.5 + np.arange(n_bins))
        
        # Subtract bare cavity frequency
        if self.fb is not None:
            cav_freq_vals -= self.fb[:,None,None]
        
        if collapse == True:
            # Collapse (seq,run) indices to give (run,bin)
            new_shape = (np.prod(cav_freq_vals.shape[:-1]),) + cav_freq_vals.shape[-1:]
            cav_freq_vals = np.reshape(cav_freq_vals, new_shape)
        
        # Choose output format
        if out_fmt == 'array':
            # Deprecated: try returning multitrace (MT)
            return( bin_times, cav_freq_vals )
        elif out_fmt == 'MT':
            return( traces.Time_Multitrace(bin_times, cav_freq_vals) )
        else:
            # Default: return MT
            return( traces.Time_Multitrace(bin_times, cav_freq_vals) )
    
    def demod_atom_trace(self, t_align):
        """
        IQ demodulates atom time traces and phase-aligns them at a specified
        time.
        
        Parameters
        ----------
        t_align : float
            Specifies what time to align trace phases at. The phase is averaged
            over a time specified in self.params.t_drive

        Returns
        -------
        atom_demod : tp.MT_Phasor
            .t : 1D np array [t]
            .V : 2D np array [run,t]
        """
        if self.atom_runs is None:
            raise Exception('atom_runs was not set!')
            
        # Collapse atom traces to (run, t) dimensions and subtract out mean
        V = np.reshape(self.atom_runs.V, 
                (np.prod(self.atom_runs.V.shape[:-1]), self.atom_runs.V.shape[-1])
                )
        V -= np.mean(V, axis=-1)[:, None]
        atom_runs_flat = traces.Time_Multitrace(self.atom_runs.t, V)
        
        # Demodulate at drive frequency (note: this is the mixed down atomic frequency)
        f0 = self.params.f0_atom
        atom_demod_raw = traces.Time_Multitrace(atom_runs_flat.t, V).iq_demod(f0)
        demod_phase = atom_demod_raw.phase()
        
        # Align phases at drive pulse
        di_ref = round(self.params.t_drive/self.params.dt)
        i0_ref = round(t_align/self.params.dt)
        phase_refs = np.mean(demod_phase.V[..., i0_ref:i0_ref+di_ref], axis=-1)
        atom_demod = traces.MT_Phasor(
            atom_demod_raw.t, atom_demod_raw.V * np.exp(-1j * phase_refs)[..., None]
            )
        
        return( atom_demod )
    
    def demod_atom_trace_OLD(self):
        """
        OLD FUNCTION. Demodulates with a fixed phase LO, loses 1/2 of the
        signal-to-noise compared to a full IQ demodulation (new function).
        """
        if self.atom_runs is None:
            raise Exception('atom_runs was not set!')

        # Collapse atom traces to (run, t) dimensions
        atom_runs_flat = traces.Time_Multitrace(self.atom_runs.t, np.reshape(self.atom_runs.V, 
            (np.prod(self.atom_runs.V.shape[:-1]), self.atom_runs.V.shape[-1])
            ))
            
        # Find the phase reference pulse (assume S/N > 1)
        trig_level = 0.6
        V = atom_runs_flat.V - np.mean(atom_runs_flat.V, axis=-1, keepdims=True)
        trig_idxs = np.argmax( 
            V > trig_level * V.max(axis=-1, keepdims=True),
            axis = -1)
        
        # Get time bin for phase reference pulse using advanced indexing
        ref_pulse_frac = 0.8
        n_bin_pts = round(self.params.t_drive / self.params.dt * ref_pulse_frac)
        ref_pulse_slices = trig_idxs[..., None] + np.arange(n_bin_pts)
        # V[[..., xk, ...], [..., yk, ...]] = [..., V[xk,yk], ...]
        ref_pulse_V = V[np.arange(V.shape[0])[:,None], ref_pulse_slices]
        ref_pulse_t = atom_runs_flat.t[ref_pulse_slices]
        
        # plt.figure()
        # for i in range(5):
        #     trace = traces.Time_Multitrace(ref_pulse_t[i,...], ref_pulse_V[i,...]) \
        #                     .fft(t_pad = self.params.t_fft_pad)
        #     plt.plot(trace.f, np.abs(trace.V)**2)
        
        """
        The fit seems like the easiest way to get phase information
        The fit doesn't return a very reliable measure of the beat frequency
        *Idea 1: don't measure frequency, assume a frequency. We need to be
            accurate to much better than 1/t_run, but if t_run is 10us
            (realistic) then we only need a frequency accuracy of ~10 kHz
            which we need anyway to probe the atoms properly in MOT stages.
        Idea 2: start with assumed frequency and find max in fft^2, binned
            to a +/- tol around f0_atom (say +/- 50 kHz). This will avoid
            AM peaks (we wouldn't notice AM peaks at 50 kHz anyway).       
        """
        # For each run, fit a frequency and phase to the reference pulse
        # FIX atomic frequency
        ref_pulse_phi = np.zeros(ref_pulse_V.shape[0])
        for r in range(ref_pulse_V.shape[0]):
            try:
                [pOpt, pCov] = curve_fit(
                        lambda t,A,phi: cos(t,A,self.params.f0_atom,phi),
                        ref_pulse_t[r,...], ref_pulse_V[r,...],
                        p0 = [1, 0]
                        )
                ref_pulse_phi[r] = pOpt[1] if pOpt[0]>0 else pOpt[1]+np.pi
            except RuntimeError:
                print(f"Fit {r+1} of {ref_pulse_V.shape[0]} failed.")
                raise
            # if r == 0:
            #     plt.figure()
            #     plt.plot(ref_pulse_t[r], ref_pulse_V[r])
            #     plt.plot(ref_pulse_t[r], cos(ref_pulse_t[r], pOpt[0], self.params.f0_atom, pOpt[1]))
            #     print(pOpt)      
        
        ref_pulse_f = np.zeros(ref_pulse_V.shape[0]) + self.params.f0_atom
        
        # Demodulate traces using fitted (f,phi) on reference pulses
        t = self.atom_runs.t[None,:]
        f, phi = ref_pulse_f[:,None], ref_pulse_phi[:,None]
        
        local_osc = cos(t, 1, f, phi)
        
        V_demod_raw = local_osc * V
        V_demod = self.params.demod_smoother(V_demod_raw)
        
        return(traces.Time_Multitrace(self.atom_runs.t, V_demod))