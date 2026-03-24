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
import glob
import os
import wer
import observation_model
import openfst_python as fst
from decoder import MyViterbiDecoder
from utils import parse_lexicon, generate_symbol_tables
import math
import time
from collections import Counter, defaultdict


def compute_unigram_probs(wav_files):
    counts = Counter()
    for wav_file in wav_files:
        transcription_file = os.path.splitext(wav_file)[0] + '.txt'
        with open(transcription_file, 'r') as f:
            words = f.readline().strip().split()
        counts.update(words)
    total = sum(counts.values())
    return {word: count / total for word, count in counts.items()}


def compute_bigram_probs(wav_files, vocab, smoothing=1):
    bigram_counts = defaultdict(Counter)
    unigram_counts = Counter()

    for wav_file in wav_files:
        transcription_file = os.path.splitext(wav_file)[0] + '.txt'
        with open(transcription_file, 'r') as f:
            words = f.readline().strip().split()
        for i in range(len(words) - 1):
            bigram_counts[words[i]][words[i + 1]] += 1
            unigram_counts[words[i]] += 1
        if words:
            unigram_counts[words[-1]] += 1

    V = len(vocab)
    bigram_probs = {}
    for w1 in vocab:
        bigram_probs[w1] = {}
        for w2 in vocab:
            count = bigram_counts[w1][w2] + smoothing
            total = unigram_counts[w1] + smoothing * V
            bigram_probs[w1][w2] = count / total

    return bigram_probs


def compute_interpolated_probs(unigram_probs, bigram_probs, vocab, lam=1.0):
    interpolated = {}
    for w1 in vocab:
        interpolated[w1] = {}
        for w2 in vocab:
            p_bigram = bigram_probs[w1][w2]
            p_unigram = unigram_probs.get(w2, 1.0 / len(vocab))
            interpolated[w1][w2] = lam * p_bigram + (1.0 - lam) * p_unigram
    return interpolated


def ensure_symbol(table, symbol):
    symbol_id = table.find(symbol)
    if symbol_id == -1:
        symbol_id = table.add_symbol(symbol)
    return symbol_id


def ensure_disambig_hmm_states(state_table, phone_symbol='#0'):
    """
    Lexicon disambiguation uses a fake phone '#0' on the phone tape. H must emit it using
    at least one HMM input id. We use a *single* pseudo-state '#0_1' only (not three), so
    decode stays cheaper than a full 3-state phone. The NN has no pdf for '#0_1';
    observation_model.log_observation_probability treats unknown labels with a neutral log-prob.
    """
    sym = f'{phone_symbol}_1'
    if state_table.find(sym) == -1:
        state_table.add_symbol(sym)


def build_L(lexicon, phone_table, word_table, add_disambig=True):
    """Lexicon transducer: phones -> words, with closure and optional #0 disambiguation."""
    L = fst.Fst('log')

    disambig_phone_id = None
    if add_disambig:
        disambig_phone_id = ensure_symbol(phone_table, '#0')
        # Word-side #0 keeps symbol tables aligned if tools inspect output alphabets.
        ensure_symbol(word_table, '#0')

    # Attach symbol tables after possible disambiguation updates.
    L.set_input_symbols(phone_table)
    L.set_output_symbols(word_table)

    start = L.add_state()
    L.set_start(start)
    # Do NOT mark start as final: that would allow an empty utterance (zero words) with
    # finite cost and can inflate insertions in WER. Only end-of-word states are finals.
    for word, phones in lexicon.items():
        current = start
        word_id = word_table.find(word)

        for i, phone in enumerate(phones):
            phone_id = phone_table.find(phone)
            next_state = L.add_state()
            out_label = word_id if i == len(phones) - 1 else 0
            L.add_arc(current, fst.Arc(phone_id, out_label, fst.Weight.One('log'), next_state))
            current = next_state

        if add_disambig:
            dis_state = L.add_state()
            L.add_arc(current, fst.Arc(disambig_phone_id, 0, fst.Weight.One('log'), dis_state))
            current = dis_state

        L.add_arc(current, fst.Arc(0, 0, fst.Weight.One('log'), start))
        L.set_final(current)

    return L


def build_G(vocab, word_table, bigram_probs, unigram_probs, lam=1.0):
    """Grammar transducer: words -> words with interpolated bigram LM weights."""
    G = fst.Fst('log')
    G.set_input_symbols(word_table)
    G.set_output_symbols(word_table)

    interpolated = compute_interpolated_probs(unigram_probs, bigram_probs, vocab, lam=lam)

    start = G.add_state()
    G.set_start(start)

    word_state = {}
    for w in vocab:
        s = G.add_state()
        word_state[w] = s
        G.set_final(s)

    for w in vocab:
        wid = word_table.find(w)
        p = max(unigram_probs.get(w, 1.0 / len(vocab)), 1e-12)
        G.add_arc(start, fst.Arc(wid, wid, fst.Weight('log', -math.log(p)), word_state[w]))

    for w1 in vocab:
        s1 = word_state[w1]
        for w2 in vocab:
            wid2 = word_table.find(w2)
            p = max(interpolated[w1][w2], 1e-12)
            G.add_arc(s1, fst.Arc(wid2, wid2, fst.Weight('log', -math.log(p)), word_state[w2]))

    return G


def build_H(phone_table, state_table, n_states=3, stay_cost=0.9):
    """Acoustic model transducer: HMM-state labels -> phone labels (incl. lexicon #0 if present)."""
    H = fst.Fst('log')
    H.set_input_symbols(state_table)
    H.set_output_symbols(phone_table)

    trans_cost = 1.0 - stay_cost
    start = H.add_state()
    H.set_start(start)
    H.set_final(start)

    # Only skip epsilon; '#0' uses a 1-state mini-HMM (see ensure_disambig_hmm_states).
    skip_phones = {'<eps>'}

    # Iterate by symbol IDs to stay compatible with openfst-python table API.
    for phone_id in range(phone_table.num_symbols()):
        p = phone_table.find(phone_id)
        if p in skip_phones:
            continue

        current = start

        # Disambiguation phone: one HMM state (#0_1) only — cheaper than a full 3-state loop.
        if p == '#0':
            state_id = state_table.find('#0_1')
            if state_id == -1:
                continue
            s1 = H.add_state()
            H.add_arc(current, fst.Arc(state_id, 0, fst.Weight('log', -math.log(trans_cost)), s1))
            H.add_arc(s1, fst.Arc(state_id, 0, fst.Weight('log', -math.log(stay_cost)), s1))
            H.add_arc(s1, fst.Arc(state_id, phone_id, fst.Weight('log', -math.log(trans_cost)), start))
            continue

        for i in range(1, n_states + 1):
            state_id = state_table.find(f"{p}_{i}")
            if state_id == -1:
                continue

            H.add_arc(current, fst.Arc(state_id, 0, fst.Weight('log', -math.log(stay_cost)), current))

            next_state = H.add_state()
            out_label = phone_id if i == n_states else 0
            H.add_arc(current, fst.Arc(state_id, out_label, fst.Weight('log', -math.log(trans_cost)), next_state))
            current = next_state

        H.add_arc(current, fst.Arc(0, 0, fst.Weight.One('log'), start))

    return H


def safe_connect(fst_obj, label='FST', verbose=True):
    """
    Remove states not on a path from start to a final state. Bindings differ:
    - some expose module-level fst.connect(ifst),
    - others in-place fst_obj.connect() on VectorFst.
    If neither works, return the graph unchanged (still valid; just may be larger).
    """
    # In-place: common in pywrapfst / openfst-python VectorFst
    if hasattr(fst_obj, 'connect'):
        try:
            fst_obj.connect()
            if verbose:
                print(f"[connect] OK (in-place): {label}")
            return fst_obj
        except Exception as e:
            if verbose:
                print(f"[connect] in-place failed {label}: {e!r}")
    try:
        fn = getattr(fst, 'connect', None)
        if fn is not None:
            out = fn(fst_obj)
            if verbose:
                print(f"[connect] OK (fst.connect): {label}")
            return out
    except Exception as e:
        if verbose:
            print(f"[connect] fst.connect failed {label}: {e!r}")
    if verbose:
        print(f"[connect] skipped: no connect() in this binding — using graph as-is ({label})")
    return fst_obj


def push_weights_safe(fst_obj, label='FST', verbose=True):
    """
    LM look-ahead: push weights toward the input (phones). If push fails, we keep the
    unstopped graph and log so experiments are not silently interpreted as 'pushed'.
    """
    try:
        fst_obj.push(push_weights=True, to_final=False)
        if verbose:
            print(f"[push] OK: {label}")
        return fst_obj
    except Exception as e1:
        try:
            out = fst.push(fst_obj, push_weights=True, to_final=False)
            if verbose:
                print(f"[push] OK (fst.push): {label}")
            return out
        except Exception as e2:
            if verbose:
                print(f"[push] FAILED {label}: {e1!r}; fallback {e2!r} — continuing without push.")
            return fst_obj


def minimize_safe(fst_obj, label='FST', verbose=True):
    try:
        fst_obj.minimize()
        if verbose:
            print(f"[minimize] OK: {label}")
        return fst_obj
    except Exception as e:
        if verbose:
            print(f"[minimize] skipped {label}: {e!r}")
        return fst_obj


def build_hlg(lexicon, vocab, word_table, phone_table, state_table, bigram_probs, unigram_probs, lam=1.0, n_states=3, stay_cost=0.9, verbose_graph_ops=True, lg_optimize=True):
    """
    lg_optimize=True: determinize + push + minimize on L o G (tree + LM look-ahead, heavier).
    lg_optimize=False: only compose L o G + connect (faster build; decode may be faster or slower
    depending on graph; use for A/B timing).
    """
    L = build_L(lexicon, phone_table, word_table, add_disambig=True)
    G = build_G(vocab, word_table, bigram_probs, unigram_probs, lam=lam)

    # After L is built, '#0' exists on the phone tape — ensure '#0_1' exists before build_H.
    ensure_disambig_hmm_states(state_table, phone_symbol='#0')

    L.arcsort(sort_type='olabel')
    G.arcsort(sort_type='ilabel')

    LG = fst.compose(L, G)
    LG = safe_connect(LG, label='L o G', verbose=verbose_graph_ops)
    if lg_optimize:
        LG_det = fst.determinize(LG)
        LG_push = push_weights_safe(LG_det, label='LG after determinize', verbose=verbose_graph_ops)
        LG_opt = minimize_safe(LG_push, label='LG after push', verbose=verbose_graph_ops)
    else:
        if verbose_graph_ops:
            print('[build_hlg] lg_optimize=False: skipping determinize / push / minimize on L o G')
        LG_opt = LG

    H = build_H(phone_table, state_table, n_states=n_states, stay_cost=stay_cost)
    H.arcsort(sort_type='olabel')
    LG_opt.arcsort(sort_type='ilabel')

    HLG = fst.compose(H, LG_opt)
    HLG = safe_connect(HLG, label='H o L o G', verbose=verbose_graph_ops)
    HLG.set_input_symbols(state_table)
    HLG.set_output_symbols(word_table)
    return HLG


def create_wfst_monolithic(lexicon, word_table, phone_table, state_table, interpolated_probs, unigram_probs, n_states=3, stay_cost=0.9):
    """Baseline: previous single-graph implementation (without silence branch)."""
    f = fst.Fst('log')
    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)

    trans_cost = 1.0 - stay_cost
    start_state = f.add_state()
    f.set_start(start_state)

    word_loop_states = {}
    for word in lexicon:
        word_loop_states[word] = f.add_state()
        f.set_final(word_loop_states[word])

    for word in lexicon:
        p_unigram = max(unigram_probs.get(word, 1.0 / len(lexicon)), 1e-12)
        f.add_arc(start_state, fst.Arc(0, 0, fst.Weight('log', -math.log(p_unigram)), word_loop_states[word]))

    for word, phones in lexicon.items():
        word_id = word_table.find(word)

        for next_word in lexicon:
            p_interp = max(interpolated_probs[word][next_word], 1e-12)
            f.add_arc(word_loop_states[word], fst.Arc(0, 0, fst.Weight('log', -math.log(p_interp)), word_loop_states[next_word]))

        current_state = word_loop_states[word]

        for phone_idx, phone in enumerate(phones):
            for i in range(1, n_states + 1):
                in_label = state_table.find(f"{phone}_{i}")
                f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight('log', -math.log(stay_cost)), current_state))

                next_state = f.add_state()
                out_label = word_id if (phone_idx == len(phones) - 1 and i == n_states) else 0
                f.add_arc(current_state, fst.Arc(in_label, out_label, fst.Weight('log', -math.log(trans_cost)), next_state))
                current_state = next_state

        f.add_arc(current_state, fst.Arc(0, 0, fst.Weight.One('log'), word_loop_states[word]))

    return f


def run_decode(f, wav_files, beam=100, max_states=None):
    """
    max_states: optional histogram pruning (keep only this many best incoming states per frame).
    Often helps H o L o G graphs with more epsilon activity; tune with WER (e.g. 60–120).
    """
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_words = 0
    total_forward_computations = 0
    total_decode_time = 0.0
    total_backtrace_time = 0.0

    for wav_file in wav_files:
        decoder = MyViterbiDecoder(f, wav_file, beam=beam, max_states=max_states)

        t0 = time.perf_counter()
        decoder.decode()
        decode_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        (state_path, words) = decoder.backtrace()
        backtrace_time = time.perf_counter() - t0

        total_decode_time += decode_time
        total_backtrace_time += backtrace_time
        total_forward_computations += decoder.forward_computation_count

        words_str = ' '.join(words)
        transcription_file = os.path.splitext(wav_file)[0] + '.txt'
        with open(transcription_file, 'r') as f_txt:
            transcription = f_txt.readline().strip()
        error_counts = wer.compute_alignment_errors(transcription, words_str)
        word_count = len(transcription.split())

        total_substitutions += error_counts[0]
        total_deletions += error_counts[1]
        total_insertions += error_counts[2]
        total_words += word_count

    overall_wer = (total_substitutions + total_deletions + total_insertions) / total_words
    return {
        'wer': overall_wer,
        'sub': total_substitutions,
        'del': total_deletions,
        'ins': total_insertions,
        'fwd': total_forward_computations,
        'decode_time': total_decode_time,
        'backtrace_time': total_backtrace_time
    }


def print_results(label, r, num_states, num_arcs):
    print(f"=== {label} ===")
    print(f"  WER                        : {r['wer']:.2%}")
    print(f"  Total substitutions        : {r['sub']}")
    print(f"  Total deletions            : {r['del']}")
    print(f"  Total insertions           : {r['ins']}")
    print(f"  Total forward computations : {r['fwd']}")
    print(f"  Total decode time          : {r['decode_time']:.4f}s")
    print(f"  Total backtrace time       : {r['backtrace_time']:.4f}s")
    print(f"  Total time                 : {r['decode_time'] + r['backtrace_time']:.4f}s")
    print(f"  WFST states                : {num_states}")
    print(f"  WFST arcs                  : {num_arcs}\n")


lex = parse_lexicon("lexicon.txt")
word_table, phone_table, state_table = generate_symbol_tables(lex)

# Decoder (MyViterbiDecoder) uses precomputed emission NLLs (ObservationModel.precompute_emission_nll)
# and cached emitting arcs per state — faster than per-arc log_observation_probability.
#
# Baseline vs HLG: for a fair WER comparison use the same beam/max_states; label any faster HLG run
# as "HLG-tuned" if you tighten beam_hlg or set max_states_hlg. HLG can have more epsilon work
# per frame; pruning knobs below target that.
# - beam_hlg < beam_baseline: often OK if LM weights are pushed (tree + lookahead).
# - max_states_hlg: histogram cap (e.g. 80–120); sweep vs WER and forward_computations.
# - lg_optimize=False: skips determinize/push/minimize on L o G (faster WFST build; A/B for decode time).
beam_baseline = 50
beam_hlg = 40
max_states_hlg = None  # e.g. 100; None = no histogram cap
lg_optimize = True

stay_cost = 0.9
lam = 1.0

wav_files = glob.glob('/group/teaching/asr/labs/recordings/*.wav')
vocab = list(lex.keys())

unigram_probs = compute_unigram_probs(wav_files)
bigram_probs = compute_bigram_probs(wav_files, vocab)
interpolated_probs = compute_interpolated_probs(unigram_probs, bigram_probs, vocab, lam=lam)

baseline_wfst = create_wfst_monolithic(
    lex, word_table, phone_table, state_table,
    interpolated_probs, unigram_probs, stay_cost=stay_cost
)

hlg_wfst = build_hlg(
    lex, vocab, word_table, phone_table, state_table,
    bigram_probs, unigram_probs, lam=lam, stay_cost=stay_cost,
    lg_optimize=lg_optimize,
)

for label, graph, beam, max_st in [
    ("Monolithic interpolated WFST", baseline_wfst, beam_baseline, None),
    ("Tree lexicon + LM look-ahead (H o L o G)", hlg_wfst, beam_hlg, max_states_hlg),
]:
    num_states = graph.num_states()
    num_arcs = sum(1 for s in graph.states() for _ in graph.arcs(s))
    result = run_decode(graph, wav_files, beam=beam, max_states=max_st)
    print_results(label, result, num_states, num_arcs)