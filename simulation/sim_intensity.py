import numpy as np
from enum import Enum

class SimulationType(Enum):
    BLINKING = 1

class SimulationParameters:
    base_counts = ...
    sd_counts = ...
    int_mod = ...
    sd_int_mod  = ...
    base_prob = ...
    sd_prob   = ...
    n_particles = ...
    n_frames: int = ...
    n_blinkers: int = ...
    
from scipy import signal

def bandpass(data, sample_rate, low=None, high=None):
    nyquist_f = sample_rate / 2

    filter_axis = len(data.shape) - 1

    if high:
        high_normal = high / nyquist_f

        b, a = signal.butter(5, high_normal, 'low', analog=False)
        filtered = signal.filtfilt(b, a, data, axis=filter_axis)

    if low:
        low_normal = low / nyquist_f

        b, a = signal.butter(5, low_normal, 'high', analog=False)
        filtered = signal.filtfilt(b, a, filtered, axis=filter_axis)

    return filtered



def simulate_intensity(parameters: SimulationParameters, sim_type: SimulationType):
    match sim_type:
        case SimulationType.BLINKING:
            transition_prob = parameters.base_prob 
            
            ## Intensity array for all frames and particles, initially filled with a random number < 1
            intensity = (np.random.random((parameters.n_particles, parameters.n_blinkers, parameters.n_frames)) <= transition_prob) * 1 

            ## Initial 50/50 chance of starting off as ON or OFF
            intensity[:, :, 0] = np.ceil((np.random.random((parameters.n_particles, parameters.n_blinkers)) > 0.5 ) * 1).astype(dtype=np.int32)    
            
            
            intensity = np.astype(intensity, np.uint32)

            I0 = np.uint32(parameters.base_counts)
            ## dI are the extra counts that an OFF trap generates
            dI = np.uint32(I0 * (parameters.int_mod - 1))

            intensity = np.cumsum(intensity, axis=2) 
            intensity %= 2
            # cumsum:
            # for i in range(1, parameters.n_frames):
            #     intensity[:, i] += intensity[:, i - 1]  
            
            ## Convert blinker ON/OFF array that contains only 0, 1 to actual counts
            ## In this representation:
            ## 0: means that the trap is ON and less PL counts will be produced
            ## 1: means thats the trap is   OFF and more PL counts will be produced

            intensity = np.sum(intensity, axis=1)

            intensity *= dI 
            intensity += I0
            
            return np.astype(intensity, np.float32)

        case _:
            raise NotImplementedError('Not implemented yet')
