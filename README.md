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
/tmp/ipykernel_169645/2249093817.py in <module>
    470     num_states = graph.num_states()
    471     num_arcs = sum(1 for s in graph.states() for _ in graph.arcs(s))
--> 472     result = run_decode(graph, wav_files, beam=beam, max_states=max_st)
    473     print_results(label, result, num_states, num_arcs)

/tmp/ipykernel_169645/2249093817.py in run_decode(f, wav_files, beam, max_states)
    375 
    376         t0 = time.perf_counter()
--> 377         decoder.decode()
    378         decode_time = time.perf_counter() - t0
    379 

~/ASR/asr_assignment/decoder.py in decode(self)
    125                              if self.V[t-1][i] < self.NLL_ZERO]
    126             active_states.sort(key=lambda x: x[1])
--> 127             allowed = set(s[0] for s in active_states[:self.max_states])
    128         else:
    129             allowed = None  # no histogram pruning

~/ASR/asr_assignment/decoder.py in forward_step(self, t)
     95             if self.V[t][i] == self.NLL_ZERO:
     96                     continue
...
--> 152         Real audio: emission_nll[t-1, pdf_idx] = -log(post_mat[t-1, pdf_idx]).
    153         OOV HMM labels (e.g. #0_1): use oov_nll = log(n_pdf) per plan.
    154         Dummy: emission_nll[t-1, col] for phonelist HMM ids plus #0_1 only. Do NOT union

KeyError: '#0_1'