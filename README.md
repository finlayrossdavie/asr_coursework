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
=== Monolithic interpolated WFST (lambda=0.0) ===
  WER                        : 47.39%
  Total substitutions        : 363
  Total deletions            : 330
  Total insertions           : 96
  Total forward computations : 10328790
  Total decode time          : 65.8134s
  Total backtrace time       : 0.0647s
  Total time                 : 65.8780s
  WFST states                : 121
  WFST arcs                  : 351

WARNING: Push: pushing type is set to 0, so not pushing
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
ERROR: State ID -1 not valid
No path got to the end of the observations.
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the observations.
ERROR: State ID -1 not valid
No path got to the end of the 


---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
/tmp/ipykernel_182432/926206671.py in <module>
    240     nh = hlg_wfst.num_states()
    241     ah = sum(1 for s in hlg_wfst.states() for _ in hlg_wfst.arcs(s))
--> 242     rh = run_decode(hlg_wfst, wav_files, beam=beam_hlg, max_states=max_states_hlg)
    243     print_results(f"Tree lexicon + LM look-ahead HLG (lambda={lam:.1f})", rh, nh, ah)
    244 

/tmp/ipykernel_182432/926206671.py in run_decode(f, wav_files, beam, max_states)
    140 
    141     for wav_file in wav_files:
--> 142         decoder = MyViterbiDecoder(f, wav_file, beam=beam, max_states=max_states)
    143 
    144         t0 = time.perf_counter()

~/ASR/asr_assignment/decoder.py in __init__(self, f, audio_file_name, beam, max_states, om)
     17 
     18         if audio_file_name:
---> 19             self.om.load_audio(audio_file_name)
     20         else:
     21             self.om.load_dummy_audio()

~/ASR/asr_assignment/observation_model.py in load_audio(self, wav_fn)
     94         self.nnet.send("{} {}\n".format(utt_name, wav_fn))
...
--> 143             return select.select(iwtd, owtd, ewtd, timeout)
    144         except InterruptedError:
    145             err = sys.exc_info()[1]

KeyboardInterrupt: 