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
Define object classes for experiment
"""
class Experiment:
    """
    Root object of a particular instance of the experiment.

    ...

    Attributes
    ----------
    has_postmeasure : bool
        Whether the sequence has the cavity probe postmeasure .
    has_premeasure : bool
        Whether the sequence has the cavity probe premeasure.
    params : Parameters (Class)
        Experimental Parameters.
    sequences: list of Sequence (Class)
        i.e. [Sequence 1, Sequence 2, Sequence 3,...]
        Each sequence corresponds to a txt file each loading sequence produces.
    

    Methods
    -------
    load_sequence(file)
        Loads a text file from a single sequence into the experiment.
    preprocess(n_seqs = None, load = 'newest', premeasure = 0,
                   premeasure_interleaved = False, postmeasure = 0,
                   n_warmups = 0, avg_fi_shots = True)
        Given loaded sequences, preprocess data contained inside and store in
        expt.data.
    """
    def __init__(self, params):
        self.sequences = []
        
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
                   premeasure_interleaved = False, postmeasure = 0,
                   n_warmups = 0, avg_fi_shots = False):
        """
        Given loaded sequences, preprocess data contained inside and return a
        Data object

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
        n_warmups : int, optional
            Specifies the number of measurement runs which should be thrown
            out. Default is 0.

        Returns
        -------
        e3l.Data object
        """
        if not self.params:
            raise Exception("Error: need to set experiment parameters!")
        if not self.sequences:
            raise Exception("Error: no sequences loaded to experiment!")
        
        ####
        # Process optional args
        ####

        # OPTIONAL ARG: n_seqs loads in a limited number of sequences
        if not n_seqs or n_seqs > len(self.sequences):
            n_seqs = len(self.sequences)
            
        # OPTIONAL ARG: load specifies in which order to load in sequences
        if load == 'oldest':
            sequences = self.sequences[:n_seqs]
        elif load == 'newest':
            sequences = self.sequences[-n_seqs:]
        else:
            sequences = self.sequences[-n_seqs:]

        # OPTIONAL ARG: override premeasure argument if interleaved specified
        if premeasure_interleaved:
            premeasure = 0
        
        ####
        # Extract data from specified sequences
        ####

        # Define a single time array for all runs of the experiment
        self.params.dt = self.sequences[0].t[1] - self.sequences[0].t[0]
        i_run = round(self.params.t_run / self.params.dt)
        t = self.params.dt * np.arange(i_run)

        # Define attributes to preprocess
        attrs = ['cav', 'atom', 'cref', 'spcm']
        attr_exists = {attr: hasattr(self.sequences[0], attr) for attr in attrs}

        # Initialize containers to sort different types of runs into
        runs = {attr: None for attr in attrs}
        preseq, postseq = {attr: None for attr in attrs}, {attr: None for attr in attrs}

        # Iterate over attributes
        for attr in attrs:
            if not attr_exists[attr]:
                continue

            # Iterate over all sequences
            for i in range(len(sequences)):
                seq = sequences[i]
                attr_MT = vars(seq)[attr] # Attribute multitrace

                # Extract runs from sequence
                if not seq.has_triggers:
                    # Assume a single run for each sequence and extract points
                    seq_runs = np.array([attr_MT.V[..., :i_run]])
                else:
                    # Edge case: redefine i_run if dataset ends abruptly
                    i_run = min(i_run, attr_MT.V.shape[-1] - seq.triggers[-1])
                    
                    # Extract runs from the single sequence using triggers as markers
                    seq_runs = np.array(
                        [attr_MT.V[..., trig:trig+i_run] for trig in seq.triggers]
                        )
                
                # Check for premeasure. preseq[attr].shape = (seq, run, t)
                if premeasure > 0 and seq_runs.shape[0] > premeasure:
                    preseq[attr] = \
                        np.concatenate((preseq[attr], [seq_runs[:premeasure]]), axis=0) \
                        if (preseq[attr] is not None) \
                        else np.array([seq_runs[:premeasure]])

                    # Remove premeasure from data run arrays    
                    seq_runs = seq_runs[premeasure:]
                
                # Check for postmeasure. postseq[attr].shape = (seq, run, t)
                if postmeasure > 0 and seq_runs.shape[0] > postmeasure:
                    postseq[attr] = \
                        np.concatenate((postseq[attr], [seq_runs[-postmeasure:]]), axis=0) \
                        if (postseq[attr] is not None) \
                        else np.array([seq_runs[-postmeasure:]])

                    # Remove postmeasure from data run arrays  
                    seq_runs = seq_runs[:-postmeasure]
                
                # Check for interleaved premeasure. preseq[attr].shape = (seq, run, t)
                if premeasure_interleaved:
                    preseq[attr] = \
                        np.concatenate((preseq[attr], [seq_runs[::2]]), axis=0) \
                        if (preseq[attr] is not None) \
                        else np.array([seq_runs[::2]])

                    # Remove interleaved premeasure from data run arrays
                    seq_runs = seq_runs[1::2]
                    
                # Add remaining runs from sequence to the full arrays. (seq, run, t)
                runs[attr] = np.concatenate((runs[attr], [seq_runs]), axis=0) \
                                if (runs[attr] is not None) \
                                else np.array([seq_runs])   

            # Throw out warmup traces
            runs[attr] = runs[attr][:, n_warmups:, :]
            if premeasure_interleaved:
                preseq[attr] = preseq[attr][:, n_warmups:, :]

        ####
        # Process premeasure and postmeasure runs
        ####
        t_bin = 0.5 * 1e-6 # 0.5 us time bins for measuring fi, fb

        # Load postmeasure into Data object
        self.has_postmeasure = (postseq['cav'] is not None)
        data_postmeasure = Data(t, postseq['cav'], postseq['atom'], self.params, cref_runs=postseq['cref']) \
                            if self.has_postmeasure else None

        # Process postmeasure. fb.shape = (seq,)
        # bare cavity frequency
        fb = None
        if self.has_postmeasure:
            fb_seqs = data_postmeasure.track_cav_frequency_iq(t_bin=t_bin, avg_sequences=False)
            fb_tbin, fb_seqs_vals = fb_seqs.t, fb_seqs.V

            # Get phasor magnitudes at t_bin values
            fb_seqs_mag = data_postmeasure._seq_cav_probe_mag()
            idx_tbin = np.round(
                    (fb_tbin - fb_seqs_mag.t0) / fb_seqs_mag.dt
                ).astype(int)
            fb_seqs_mag2_vals = fb_seqs_mag.V[..., idx_tbin]**2

            # Calculate single fb for each sequence
            fb = np.average(fb_seqs_vals, axis=-1, weights=fb_seqs_mag2_vals)

        # Load premeasure into Data object
        self.has_premeasure = (preseq['cav'] is not None)
        data_premeasure = Data(t, preseq['cav'], preseq['atom'], self.params, fb=fb, cref_runs=preseq['cref']) \
                            if self.has_premeasure else None
        
        # Process premeasure. fi.shape = (seq,)
        # initial cavity frequency
        fi = None
        if self.has_premeasure:
            fi_seqs = data_premeasure.track_cav_frequency_iq(t_bin=t_bin, avg_sequences=False, use_cref=avg_fi_shots, avg_shots=avg_fi_shots)
            fi_tbin, fi_seqs_vals = fi_seqs.t, fi_seqs.V

            # Get phasor magnitudes at t_bin values
            fi_seqs_mag = data_premeasure._seq_cav_probe_mag(avg_shots=avg_fi_shots)
            idx_tbin = np.round(
                    (fi_tbin - fi_seqs_mag.t0) / fi_seqs_mag.dt
                ).astype(int)
            fi_seqs_mag2_vals = fi_seqs_mag.V[..., idx_tbin]**2

            # Calculate single fi for each sequence (or array of shots if avg_shots = False)
            fi = np.average(fi_seqs_vals, axis=-1, weights=fi_seqs_mag2_vals)

        # Assign to data object
        data = Data(t, runs['cav'], runs['atom'], self.params, fi=fi, fb=fb, \
                            cref_runs=runs['cref'], spcm_runs=runs['spcm'])
            
        return( data, data_premeasure, data_postmeasure )
            
class Parameters:
    """
    Contains the experimental parameters

    ...

    Attributes
    ----------
    t_run : float
        Length of each shot (s)
    t_drive : float
        Duration of Rabi drive (s)
    t_cav_pulse : float
        Cycle time of cavity probe, i.e. the cavity is whacked every 
        t_cav_pulse (s).
    f0_cav : float
        Effective "osillation" frequency (Hz) that oscilloscope acquires for 
        cavity probe.
    f0_atom : float
        Effective "osillation" frequency (Hz) that oscilloscope acquires for 
        atomic probe.
    dt : float  
        Time difference (s) between each data point that scope samples.
    Methods
    -------
    None.
    """
    def __init__(self):
        self.t_run = None
        self.t_drive = None
        self.t_cav_pulse = None
        self.f0_cav = None
        self.f0_atom = None
        self.dt = None
        
    def _all_params_defined(self):
        for attr in vars(self):
            if vars(self)[attr] is None:
                return(False)
        return(True)

class Data:
    """
    Stores preprocessed data. Accessed via the parent Experiment object. 

    ...

    Attributes
    ----------
    atom_runs : Time_Multitrace (Class)
        Atomic probe data.
    cav_runs : Time_Multitrace (Class)
        Cavity probe data.
    cref_runs : Time_Multitrace (Class)
        RF reference tone data [for calibration of cavity probe phase].
    fb : ndarray of shape (seq,)
        Bare cavity frequency.
    fi : ndarray of shape (seq,)
        Initial cavity frequency with all atoms in the ground states.
    params : Parameters (Class)
        Experimental Parameters.
    spcm_runs : Time_Multitrace (Class)
        TO BE IMPLEMENTED.
    t : ndarray
        Time coodinates of dynamics of one shot.

    
    -------
    subset(idx)
        Returns a Data object which is sliced along the sequence axis with
        the given 1D numpy array idx.

    track_cav_frequency_iq(f_demod = None, align = True, avg_sequences = True,
                               ignore_pulse_bins = True, use_cref = True, 
                               avg_shots = True)
        IQ demodulates cavity time traces, bins them, and fits their phase(t)
        with a linear regression to estimate instantaneous frequency. Multiple
        shots give statistics on these bins.

    demod_atom_trace(t_align=None, collapse=True)
        IQ demodulates atom time traces and phase-aligns them at a specified
        time.

    avg_spcm_traces
        NOT IMPLEMENTED.

    """
    def __init__(self, t, cav_runs, atom_runs, params, fi=None, fb=None, cref_runs=None, spcm_runs=None):
        self.t = t
        self.cav_runs = traces.Time_Multitrace(t, cav_runs) \
            if (cav_runs is not None) else None
        self.atom_runs = traces.Time_Multitrace(t, atom_runs) \
            if (atom_runs is not None) else None
        self.params = params
        
        # Initial and bare cavity frequency estimators. Shape: (seq,)
        self.fi = fi
        self.fb = fb

        # Cavity phase reference data
        self.cref_runs = traces.Time_Multitrace(t, cref_runs) \
            if (cref_runs is not None) else None

        self.spcm_runs = traces.Time_Multitrace(t, spcm_runs) \
            if (spcm_runs is not None) else None

    def _seq_cav_probe_mag(self, f_demod = None, avg_shots=True):
        """
        Compute the amplitude of the demodulated phasor (Time_Multitrace).
        Returns in (seq, t) if avg_shots is True, and (seq, run, t) otherwise.

        Parameters
        ----------
        f_demod : float, optional
            Frequency to demodulate signal by. Units are period units in t. 
        avg_shots : boolean, optional
            Specifies whether or not to compute the average of the magnitude.
        """
        if self.cav_runs is None:
            raise Exception('cav_runs was not set!')
        
        # OPTIONAL ARG: Pull f_demod if not specified
        if f_demod is None:
            f_demod = self.params.f0_cav
            
        # Demodulate cavity data
        cav_runs = self.cav_runs
        cav_runs.V -= np.mean(cav_runs.V, axis=-1, keepdims=True)
        cav_runs_mag = cav_runs.iq_demod(f_demod).mag()

        # Average over runs. (seq, run, t) -> (seq, t)
        if avg_shots:
            cav_seqs_mag = traces.Time_Multitrace(cav_runs_mag.t,
                    np.average(cav_runs_mag.V, axis=1)
                )
        else:
            cav_seqs_mag = cav_runs_mag
    
        return(cav_seqs_mag)

    def subset(self, idx):
        """
        Returns a Data object which is sliced along the sequence axis with
        the given 1D numpy array idx.

        Parameters
        ----------
        idx : ndarray
            Selects which sequences to keep.
        """
        idx_full = lambda dim: (idx,) + (slice(None),) * (dim-1)

        new_atom_runs = self.atom_runs.V[idx_full(self.atom_runs.dim)]
        new_cav_runs = self.cav_runs.V[idx_full(self.cav_runs.dim)]

        if self.cref_runs is not None:
            new_cref_runs = self.cref_runs.V[idx_full(self.cref_runs.dim)]
        else:
            new_cref_runs = None

        if self.fi is not None:
            new_fi = self.fi[ idx_full(self.fi.ndim) ]
        else:
            new_fi = None

        if self.fb is not None:
            new_fb = self.fb[ idx_full(self.fb.ndim) ]
        else:  
            new_fb = None

        return( Data(self.t, new_cav_runs, new_atom_runs, self.params, fi=new_fi, fb=new_fb, cref_runs=new_cref_runs) )

    
    def track_cav_frequency_iq(self, t_bin, f_demod = None, align = True, avg_sequences = True, \
                               ignore_pulse_bins = True, use_cref = True, avg_shots = True):
        """
        IQ demodulates cavity time traces, bins them, and fits their phase(t)
        with a linear regression to estimate instantaneous frequency. Multiple
        shots give statistics on these bins.
        
        Parameters
        ----------
        t_bin : float
            Specifies the size of time bins for measuring the cavity frequency.
        f_demod : float, optional
            Specifies what frequency to demodulate at. If None, f_demod is auto-set
            to self.params.f0_cav. The default is None.
        align : boolean, optional
            Specifies whether or not to align the time trace bins to a potential
            pulsed cavity probe. If True, looks for the time 0 <= t < t_bin
            corresponding to the maximum value of the first cavity trace and
            truncates the trace to start at this time, then bins. If False,
            performs no truncation. Default is True.
        avg_sequences : boolean, optional
            Specifies whether to average frequency measurements across sequences
            in the experiment. If True, returns a Time_MT with dimension [tbin]
            along with a standard deviation of the mean across sequences in dV.
            If False, returns a Time_MT with dimension [seq, tbin] with a NoneType
            value for dV.
        ignore_pulse_bins : boolean, optional
            Specifies whether to ignore time bins that occur during a pulse.
            Assumes an integer number of t_bins in a t_cav_pulse. Default is
            True.
        use_cref : boolean, optional
            Specifies whether to apply phase corrections to the cavity phasors
            using the experiment's cavity reference RF data, if it exists.
            Default is True.

        Returns
        -------
        cav_freqs : tp.Time_Multitrace
            .t : 1D np array [bin]
            .V : 1D np array [bin] or 2D np array [seq,bin] depending on avg_sequences
            .dV : 1D np array [bin] or None depending on avg_sequences
        """
        if self.cav_runs is None:
            raise Exception('cav_runs was not set!')
        
        # OPTIONAL ARG: Pull f_demod if not specified
        if f_demod is None:
            f_demod = self.params.f0_cav
            
        # Demodulate cavity data
        cav_runs = self.cav_runs
        cav_runs.V -= np.mean(cav_runs.V, axis=-1, keepdims=True)
        cav_phasor_raw = cav_runs.iq_demod(f_demod)

        # Check for cavity phase reference trace. If it exists, align cavity phasors with this data
        if (self.cref_runs is not None) and use_cref:
            cref_phasor = self.cref_runs.iq_demod(f_demod)
            cref_mag2_raw = cref_phasor.mag().V**2
            cref_mag2_max = np.max(cref_mag2_raw, axis=-1)
            threshold_filter = (cref_mag2_raw > 0.5 * cref_mag2_max[...,None]).astype(int)
            cref_mag2 = threshold_filter * cref_mag2_raw # small magnitudes set to 0

            cref_phase = cref_phasor.phase(unwrap=False)

            # Calcluate weighted average, allowing for weights to be different between shots
            y, w = cref_phase.V, cref_mag2
            cref_avg_phase = np.sum(y*w, axis=-1) / np.sum(w, axis=-1)

            # Correct cavity phasors and average phasors within single sequences
            cav_phasor_corr_V = cav_phasor_raw.V * np.exp(-1j * cref_avg_phase[..., None])
            seq_phasor_avg = np.average(cav_phasor_corr_V, axis=-2)
            seq_phasor_stdmean = np.std(cav_phasor_corr_V, axis=-2)/np.sqrt(cav_phasor_corr_V.shape[-2])
            cav_phasor = traces.MT_Phasor(cav_phasor_raw.t,
                                          seq_phasor_avg, dV=seq_phasor_stdmean)
        else:
            use_cref = False
            cav_phasor = cav_phasor_raw

        cav_phase = cav_phasor.phase()
        phase = cav_phase.V
        
        t_pulse = self.params.t_cav_pulse
        n_pulse_pts = round( t_pulse/cav_phase.dt )
        n_pts = phase.shape[-1]
        n_runs = np.prod(phase.shape[:-1])        
        
        # Which is larger: cav pulse time spacing or time bin length?
        if int(self.params.t_cav_pulse/t_bin) > 1:
            n_bin_pts = round( t_bin/cav_phasor.dt )

            # OPTIONAL ARG: Align to pulses
            # DELETED CODE [see v0.5.0]: option use_phase_jumps = False (instead aligns to max voltage in pulse)
            if align and not use_cref:
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
                
                cav_phasor = cav_phasor.truncate(t_min = cav_phasor.t[i0])
            elif align and use_cref:
                # In average magnitude array, find peak and trough of a pulse
                mag = np.average(cav_phasor.mag().V, axis=0)
                i_pulse_max = n_pulse_pts + np.argmax(mag[n_pulse_pts:])
                i_pulse_min = i_pulse_max-n_pulse_pts + np.argmin(mag[i_pulse_max-n_pulse_pts:i_pulse_max])
                
                # Define i0_pulse to be halfway between min and max
                i0_pulse = int((i_pulse_max + i_pulse_min)/2) % n_pulse_pts
                i0 = i0_pulse + int(n_bin_pts/2)
                
                cav_phasor = cav_phasor.truncate(t_min = cav_phasor.t[i0])
            
            # Bin cavity phasor traces
            cav_phase = cav_phasor.phase()
            cav_bins = cav_phase.bin_trace(t_bin)
                                    
            # Get bin times
            bin_times = cav_phasor.t[0] + t_bin * (0.5 + np.arange(cav_bins.V.shape[-2]))
            
            # Estimate cavity frequency in bins using linear regression
            cav_freq_vals = cav_bins.frequency()
            
            # OPTIONAL ARG: Remove pulse bins
            if ignore_pulse_bins and align:                            
                # Will ignore every [n_pulse_bins]-th bin
                n_pulse_pts = round( self.params.t_cav_pulse/cav_phasor.dt )
                n_pulse_bins = round( n_pulse_pts / n_bin_pts )
                
                # Delete offending bins
                # pulse_slice = slice(n_pulse_bins-1, None, n_pulse_bins)
                pulse_slice = slice(n_pulse_bins-1, None, n_pulse_bins)
                cav_freq_vals = np.delete(cav_freq_vals, pulse_slice, axis=-1)
                bin_times = np.delete(bin_times, pulse_slice, axis=-1)

        else:
            # Calculate frequency in bins of each pulse, since phase jumps seem to occur during pulses
            # DELETED CODE [see v0.5.0]: options for runs_aligned = False, single_index_align = True            

            # Collapse into trace of duration t_pulse
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

            # Construct phase object with aligned pulses
            t_aligned = cav_phase.t[:n_pulses*n_pulse_pts]
            phase_flat_aligned = traces.MT_Phase(t_aligned, phase_flat_aligned_vals)

            wfunc = lambda x: np.sin(np.pi * np.cos(np.pi/2*(1-x)) )**2
            # wfunc = None
            cav_freq_vals_flat = phase_flat_aligned.frequency(t_bin=t_pulse, wfunc=wfunc)
            
            n_bin_pulses = int(t_bin/t_pulse)
            n_bins = int(n_pulses/n_bin_pulses)
            cav_freq_vals_bins = cav_freq_vals_flat[:,:n_bins*n_bin_pulses] \
                .reshape((n_runs, n_bins, n_bin_pulses)) \
                .mean(axis=-1)

            # Shape array of frequency values back to (seq, run, t_bin)          
            cav_freq_vals = cav_freq_vals_bins \
                .reshape(phase.shape[:-1] + (n_bins,))
            bin_times = cav_phase.t[0] + t_bin * (0.5 + np.arange(n_bins))
        
        # If no cavity reference exists, average runs within a sequence
        if not use_cref:
            if avg_shots:
                # Average (seq, run, t) to (seq, t)
                cav_freq_vals = np.average(cav_freq_vals, axis=1)

        # Subtract bare cavity frequency
        if self.fb is not None:
            cfv_ndim = cav_freq_vals.ndim
            cav_freq_vals -= self.fb[(..., *([None]*(cfv_ndim-1)) )]
                
        # Build Time_MT object
        cav_freqs = traces.Time_Multitrace(bin_times, cav_freq_vals)
        
        # OPTIONAL ARG: average frequency measurements over sequences
        if avg_sequences is True:
            freqs_mean = np.average(cav_freqs.V, axis=0)
            freqs_stdmean = np.std(cav_freqs.V, axis=0)/np.sqrt(cav_freqs.V.shape[0])

            cav_freqs = traces.Time_Multitrace(bin_times,
                                freqs_mean, dV=freqs_stdmean)
        
        # Return Time_MT object
        return( cav_freqs )
    
    def demod_atom_trace(self, t_align=None, collapse=True):
        """
        IQ demodulates atom time traces and phase-aligns them at a specified
        time.
        
        Parameters
        ----------
        t_align : float, optional.
            Specifies what time to align trace phases at. The phase is averaged
            over a time specified in self.params.t_drive. If None, function does
            not perform phase alignment. Default is None.

        collapse : bool, optional
            Specifies whether or not to flatten the non-time dimensions of the
            multitrace. Default is True.

        Returns
        -------
        atom_demod : tp.MT_Phasor
            .t : 1D np array [t]
            .V : 2D np array [run,t]
        """
        if self.atom_runs is None:
            raise Exception('atom_runs was not set!')
            
        # Subtract out mean
        V = self.atom_runs.V - np.mean(self.atom_runs.V, axis=-1)[..., None]


        # Collapse dimensions, then IQ demodulate at the mixed down atomic frequency
        f0 = self.params.f0_atom
        if collapse:
            atom_demod_raw = traces.Time_Multitrace(self.atom_runs.t, V) \
                                .collapse() \
                                .iq_demod(f0)
        else:
            atom_demod_raw = traces.Time_Multitrace(self.atom_runs.t, V) \
                                .iq_demod(f0)            
        demod_phase = atom_demod_raw.phase()
        
        # Align phases at drive pulse
        if t_align is not None:
            di_ref = round(self.params.t_drive/self.params.dt)
            i0_ref = round(t_align/self.params.dt)
            phase_refs = np.mean(demod_phase.V[..., i0_ref:i0_ref+di_ref], axis=-1)
            atom_demod = traces.MT_Phasor(
                atom_demod_raw.t, atom_demod_raw.V * np.exp(-1j * phase_refs)[..., None]
                )
        else:
            atom_demod = atom_demod_raw
        
        return( atom_demod )

    def avg_spcm_traces(self):
        if self.spcm_runs is None:
            raise Exception('spcm_runs was not set!')

        return( self.spcm_runs.average_over(dim=1).average_over(dim=0) )