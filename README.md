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

# ==========================================
# 1. PASTE LAB 3/4 HELPER FUNCTIONS
# ==========================================
def parse_lexicon(lex_file):
    lex = {}  
    with open(lex_file, 'r') as f:
        for line in f:
            line = line.split()  
            lex[line[0]] = line[1:]  
    return lex

def generate_symbol_tables(lexicon, n=3):
    state_table = fst.SymbolTable()
    phone_table = fst.SymbolTable()
    word_table = fst.SymbolTable()
    
    state_table.add_symbol('<eps>')
    phone_table.add_symbol('<eps>')
    word_table.add_symbol('<eps>')
    
    for word, phones in lexicon.items():
        word_table.add_symbol(word)
        for p in phones:
            phone_table.add_symbol(p)
            for i in range(1,n+1): 
                state_table.add_symbol('{}_{}'.format(p, i))
    return word_table, phone_table, state_table

# Load the lexicon and generate global symbol tables as done in the labs
lex = parse_lexicon('lexicon.txt')
word_table, phone_table, state_table = generate_symbol_tables(lex)

def generate_phone_wfst(f, start_state, phone, n):
    current_state = start_state
    for i in range(1, n+1):
        in_label = state_table.find('{}_{}'.format(phone, i))
        sl_weight = fst.Weight('log', -math.log(0.1)) 
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))
        
        if i == n:
            out_label = phone_table.find(phone)
        else:
            out_label = 0  
            
        next_state = f.add_state()
        next_weight = fst.Weight('log', -math.log(0.9)) 
        f.add_arc(current_state, fst.Arc(in_label, out_label, next_weight, next_state))    
        current_state = next_state
        
    return current_state

# ==========================================
# 2. DEFINE create_wfst()
# ==========================================
def create_wfst():
    """
    Creates an overarching WFST for the decoding task.
    This example creates a simple word loop allowing any word sequence.
    """
    f = fst.Fst('log')
    start_state = f.add_state()
    f.set_start(start_state)
    
    # Loop back state allows transitioning between words
    loop_state = f.add_state()
    f.add_arc(start_state, fst.Arc(0, 0, fst.Weight('log', 0.0), loop_state))
    f.set_final(loop_state)
    
    # Connect every word in the lexicon to the loop
    for word in lex.keys():
        current_state = loop_state
        
        # Add phones for the word
        for phone in lex[word]:
            current_state = generate_phone_wfst(f, current_state, phone, 3)
            
        # Tie the end of the word back to the loop state
        # Output the actual word label at the end of the word path
        out_label = word_table.find(word)
        f.add_arc(current_state, fst.Arc(0, out_label, fst.Weight('log', 0.0), loop_state))
        
    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)
    return f

# ==========================================
# 3. PASTE MyViterbiDecoder FROM LAB 3/4
# ==========================================
class MyViterbiDecoder:
    # (Paste the FULL MyViterbiDecoder class from asr_lab3_4_solutions.ipynb here)
    # Ensure __init__, initialise_decoding, traverse_epsilon_arcs, forward_step,
    # finalise_decoding, decode, and backtrace methods are all included exactly 
    # as they appear in the lab 3/4 solutions.
    pass

# ==========================================
# 4. ASSIGNMENT EXECUTION & WER ACCUMULATION
# ==========================================
def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """
    transcription_file = os.path.splitext(wav_file)[0] + '.txt'
    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()
    return transcription

f_wfst = create_wfst()

# Initialize accumulators for WER
total_substitutions = 0
total_deletions = 0
total_insertions = 0
total_words = 0

# Iterate through the files
for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'): 
    
    decoder = MyViterbiDecoder(f_wfst, wav_file)
    decoder.decode()
    (state_path, words) = decoder.backtrace() 
    
    transcription = read_transcription(wav_file)
    error_counts = wer.compute_alignment_errors(transcription, words)
    word_count = len(transcription.split())
    
    # error_counts is a tuple: (substitutions, deletions, insertions)
    total_substitutions += error_counts[0]
    total_deletions += error_counts[1]
    total_insertions += error_counts[2]
    total_words += word_count
    
    print(f"File: {os.path.basename(wav_file)} | Errors (S,D,I): {error_counts} | Words: {word_count}")

# Compute overall WER
overall_wer = (total_substitutions + total_deletions + total_insertions) / total_words
print(f"\nOverall Word Error Rate (WER): {overall_wer:.2%}")

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
