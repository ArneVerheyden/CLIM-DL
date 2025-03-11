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
    ### Constant in the PLQY function
    quencher_strength: float = ...
    
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
            blinkers = (np.random.random((parameters.n_particles, parameters.n_blinkers, parameters.n_frames)) <= transition_prob) * 1 

            ## Initial 50/50 chance of starting off as ON or OFF
            blinkers[:, :, 0] = ((np.random.random((parameters.n_particles, parameters.n_blinkers)) > 0.5 ) * 1).astype(dtype=np.uint32)    

            I0 = np.uint32(parameters.base_counts)

            blinkers = np.cumsum(blinkers, axis=2) 
            blinkers %= 2
            # cumsum:
            # for i in range(1, parameters.n_frames):
            #     intensity[:, i] += intensity[:, i - 1]  
            
            ## Convert blinker ON/OFF array that contains only 0, 1 to actual counts
            ## In this representation:
            ## 0: means that the trap is ON and less PL counts will be produced
            ## 1: means thats the trap is   OFF and more PL counts will be produced
            active_blinkers = np.sum(blinkers, axis=1)
            intensity = I0 / (1 + parameters.quencher_strength * active_blinkers)

            return np.astype(intensity, np.float32)

        case _:
            raise NotImplementedError('Not implemented yet')
