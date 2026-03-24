# ASR Assignment 2026

Code for the assignment of the ASR course for 2025/26.  Please do not make a public fork of this repo, as you may not share code outside of your pair.

Additional resources not in this repository:
* [Main assignment document (pdf)](https://opencourse.inf.ed.ac.uk/sites/default/files/https/opencourse.inf.ed.ac.uk/asr/2026/assignment.pdf)
* [Recording interface](https://homepages.inf.ed.ac.uk/gkarakas/recorder.html)
* [Consent form](https://github.com/yiwang454/asr_assignment/blob/master/consent_form/labs_consent_form.pdf) ; [MS Form Version](https://forms.office.com/e/mcyCDv3hCu)
* [Participant Information Sheet](https://github.com/yiwang454/asr_assignment/blob/master/consent_form/labs_pis_2526.pdf)


import glob
import os
import wer
import observation_model
import math
import openfst_python as fst


# Setup 

You can follow the same setup as for the labs, https://github.com/geoph9/asr_labs

You can clone this repository and open the notebook with the following commands:

```shell
git clone https://github.com/yiwang454/asr_assignment.git

# Load your original lab virtual environment
source /opt/conda/etc/profile.d/conda.sh
conda activate asr_env

# Open the assignment notebook
cd asr_assignment
jupyter notebook 
```

---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_169645/2522638963.py in <module>
    452     num_states = graph.num_states()
    453     num_arcs = sum(1 for s in graph.states() for _ in graph.arcs(s))
--> 454     result = run_decode(graph, wav_files, beam=beam, max_states=max_st)
    455     print_results(label, result, num_states, num_arcs)

/tmp/ipykernel_169645/2522638963.py in run_decode(f, wav_files, beam, max_states)
    363 
    364         t0 = time.perf_counter()
--> 365         decoder.decode()
    366         decode_time = time.perf_counter() - t0
    367 

~/ASR/asr_assignment/decoder.py in decode(self)
    125                             self.V[t][j] = prob
    126                             self.B[t][j] = i
--> 127 
    128                             if arc.olabel != 0:
    129                                 self.W[t][j] = [arc.olabel]

~/ASR/asr_assignment/decoder.py in forward_step(self, t)
     95         if self.max_states is not None:
     96             active_states = [(i, self.V[t-1][i]) for i in self._state_ids
...
--> 152         if self.dummy:
    153             return self.dummy_observation_probability(hmm_label, t)
    154 

KeyError: '#0_1'