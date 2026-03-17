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


NOTES: 

def create_wfst():
    # 1. Build lexicon and symbol tables
    lex = parse_lexicon('lexicon.txt')
    word_table, phone_table, state_table = generate_symbol_tables(lex, n=3)
    f = fst.Fst('log')
    # 2. Single word-boundary state
    start_state = f.add_state()
    f.set_start(start_state)
    # 3. For each word, build an HMM from the shared boundary state back to it
    for word, phones in lex.items():
        current_state = start_state
        for idx, phone in enumerate(phones):
            is_last_phone = (idx == len(phones) - 1)
            if is_last_phone:
                word_label = word_table.find(word)
                current_state = generate_phone_wfst(
                    f, current_state, phone, 3,
                    state_table,
                    word_table,
                    output_is_word=True,
                    out_label=word_label
                )
            else:
                current_state = generate_phone_wfst(
                    f, current_state, phone, 3,
                    state_table,
                    phone_table,
                    output_is_word=False
                )
        # After last phone of word, loop back to boundary with epsilon
        loop_weight = fst.Weight('log', 0.0)  # log(1.0)
        f.add_arc(current_state, fst.Arc(0, 0, loop_weight, start_state))
    # 4. Make the boundary state final (can end after any word sequence)
    f.set_final(start_state)
    # 5. Attach symbol tables
    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)
    return f
