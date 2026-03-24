"""
Build H ∘ (L ∘ G): phone HMM, tree lexicon, bigram LM with optional determinize/push/minimize on L∘G.
"""
import math
from collections import defaultdict

import openfst_python as fst


def safe_connect(fst_obj, label=""):
    """Connect to trim unreachable states; in-place if available."""
    try:
        if hasattr(fst_obj, "connect"):
            fst_obj.connect()
    except Exception as e:
        if label:
            print(f"[safe_connect] {label}: {e}")


def push_weights_safe(fst_obj, label=""):
    try:
        return fst.push(fst_obj)
    except Exception as e:
        if label:
            print(f"[push_weights_safe] {label}: {e}")
        return fst_obj


def minimize_safe(fst_obj, label=""):
    try:
        return fst.minimize(fst_obj)
    except Exception as e:
        if label:
            print(f"[minimize_safe] {label}: {e}")
        return fst_obj


def add_phone_hmm_chain(
    f,
    start_state,
    phone,
    n_states,
    stay_cost,
    trans_cost,
    phone_table,
    state_table,
):
    """Left-to-right phone HMM; emit phone id on last forward transition."""
    current_state = start_state
    for i in range(1, n_states + 1):
        state_sym = f"{phone}_{i}"
        in_label = state_table.find(state_sym)
        sl_weight = fst.Weight("log", -math.log(stay_cost))
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))
        next_state = f.add_state()
        fw_weight = fst.Weight("log", -math.log(trans_cost))
        if i == n_states:
            out_label = phone_table.find(phone)
        else:
            out_label = 0
        f.add_arc(current_state, fst.Arc(in_label, out_label, fw_weight, next_state))
        current_state = next_state
    return current_state


def build_H(phone_table, state_table, phones, n_states=3, stay_cost=0.9, n_sil_states=5):
    """
    Single pronunciation HMM FST: input = HMM pdf ilabels, output = phone ids.
    From a hub state, epsilon enters each phone chain or silence; chains return to hub.
    """
    trans_cost = 1.0 - stay_cost
    f = fst.Fst("log")
    f.set_input_symbols(state_table)
    f.set_output_symbols(phone_table)

    hub = f.add_state()
    f.set_start(hub)

    sil_states = []
    for _ in range(n_sil_states):
        sil_states.append(f.add_state())
    for i, sil_state in enumerate(sil_states):
        sil_label = state_table.find(f"sil_{i+1}")
        f.add_arc(
            sil_state,
            fst.Arc(sil_label, 0, fst.Weight("log", -math.log(stay_cost)), sil_state),
        )
        if i < n_sil_states - 1:
            f.add_arc(
                sil_state,
                fst.Arc(
                    sil_label,
                    0,
                    fst.Weight("log", -math.log(trans_cost)),
                    sil_states[i + 1],
                ),
            )
        else:
            f.add_arc(
                sil_state,
                fst.Arc(
                    sil_label,
                    0,
                    fst.Weight("log", -math.log(trans_cost)),
                    hub,
                ),
            )

    f.add_arc(hub, fst.Arc(0, 0, fst.Weight.One(f.weight_type()), sil_states[0]))

    for phone in sorted(phones):
        if phone in ("<eps>",):
            continue
        try:
            pid = phone_table.find(phone)
        except Exception:
            continue
        if pid == 0:
            continue
        entry = f.add_state()
        f.add_arc(hub, fst.Arc(0, 0, fst.Weight.One(f.weight_type()), entry))
        last = add_phone_hmm_chain(
            f,
            entry,
            phone,
            n_states,
            stay_cost,
            trans_cost,
            phone_table,
            state_table,
        )
        f.add_arc(last, fst.Arc(0, 0, fst.Weight.One(f.weight_type()), hub))

    return f


def build_L(lexicon, word_table, phone_table, add_disambig=True):
    """
    Prefix tree lexicon: input = phone ids, output = word id on last phone of each word.
    Homophones (same pronunciation): second+ words get an extra #0 phone if add_disambig.
    """
    f = fst.Fst("log")
    f.set_input_symbols(phone_table)
    f.set_output_symbols(word_table)

    if add_disambig:
        try:
            phone_table.find("#0")
        except Exception:
            try:
                phone_table.add_symbol("#0")
            except Exception:
                pass

    root = f.add_state()
    f.set_start(root)
    children = {}

    pronunciation_map = defaultdict(list)
    for word, phones in lexicon.items():
        pronunciation_map[tuple(phones)].append(word)

    # Shorter pronunciations first so a word that is a prefix of another gets its final arc
    # before the longer path extends (e.g. "a" ah before "of" ah v).
    sorted_tuples = sorted(pronunciation_map.keys(), key=lambda t: len(t))
    for phones_tuple in sorted_tuples:
        words = pronunciation_map[phones_tuple]
        words_sorted = sorted(words)
        for wi, word in enumerate(words_sorted):
            phones = list(phones_tuple)
            if wi > 0 and add_disambig:
                phones.append("#0")
            cur = root
            wid = word_table.find(word)
            for idx, ph in enumerate(phones):
                pid = phone_table.find(ph)
                key = (cur, pid)
                is_last = idx == len(phones) - 1
                olabel = wid if is_last else 0
                if key not in children:
                    ns = f.add_state()
                    children[key] = ns
                    f.add_arc(
                        cur,
                        fst.Arc(pid, olabel, fst.Weight.One(f.weight_type()), ns),
                    )
                    cur = ns
                else:
                    cur = children[key]
            f.set_final(cur)

    return f


def build_G(vocab, word_table, interpolated_probs, unigram_probs):
    """
    Bigram LM with interpolation: first word from unigram; transitions use interpolated_probs[w1][w2].
    """
    f = fst.Fst("log")
    f.set_input_symbols(word_table)
    f.set_output_symbols(word_table)

    s0 = f.add_state()
    f.set_start(s0)

    ctx = {}
    for w in vocab:
        ctx[w] = f.add_state()

    V = len(vocab)
    for w in vocab:
        wid = word_table.find(w)
        pu = unigram_probs.get(w, 1.0 / V)
        f.add_arc(
            s0,
            fst.Arc(wid, 0, fst.Weight("log", -math.log(pu)), ctx[w]),
        )

    for w1 in vocab:
        s1 = ctx[w1]
        for w2 in vocab:
            p = interpolated_probs[w1][w2]
            wid2 = word_table.find(w2)
            f.add_arc(
                s1,
                fst.Arc(wid2, 0, fst.Weight("log", -math.log(p)), ctx[w2]),
            )

    for w in vocab:
        f.set_final(ctx[w])

    return f


def ensure_disambig_hmm_states(state_table, phone_table, phone="#0", n_states=3):
    """Ensure phone and hmm_1..n_states exist for disambiguation phone."""
    try:
        phone_table.find(phone)
    except Exception:
        try:
            phone_table.add_symbol(phone)
        except Exception:
            pass
    for i in range(1, n_states + 1):
        sym = f"{phone}_{i}"
        try:
            state_table.find(sym)
        except Exception:
            state_table.add_symbol(sym)


def build_hlg(
    lexicon,
    vocab,
    word_table,
    phone_table,
    state_table,
    interpolated_probs,
    unigram_probs,
    n_states=3,
    stay_cost=0.9,
    lg_optimize=True,
    add_disambig=True,
    verbose_graph_ops=True,
):
    """
    Compose H ∘ (L ∘ G); optionally determinize + push + minimize on L ∘ G.
    """
    phone_sets = [set(p) for p in lexicon.values()]
    phones_for_h = set.union(*phone_sets, {"sil"}) if phone_sets else {"sil"}
    if add_disambig:
        ensure_disambig_hmm_states(state_table, phone_table, "#0", n_states=n_states)
        phones_for_h.add("#0")

    H = build_H(
        phone_table,
        state_table,
        phones_for_h,
        n_states=n_states,
        stay_cost=stay_cost,
    )
    L = build_L(lexicon, word_table, phone_table, add_disambig=add_disambig)
    G = build_G(vocab, word_table, interpolated_probs, unigram_probs)

    L.arcsort(sort_type="olabel")
    G.arcsort(sort_type="ilabel")
    LG = fst.compose(L, G)
    safe_connect(LG, label="L o G" if verbose_graph_ops else "")

    if lg_optimize:
        try:
            LG_det = fst.determinize(LG)
        except Exception as e:
            if verbose_graph_ops:
                print(f"[build_hlg] determinize failed, using connected L o G: {e}")
            LG_det = LG
        LG_push = push_weights_safe(LG_det, label="LG after determinize" if verbose_graph_ops else "")
        LG_opt = minimize_safe(LG_push, label="LG after push" if verbose_graph_ops else "")
    else:
        if verbose_graph_ops:
            print("[build_hlg] lg_optimize=False: skipping determinize / push / minimize on L o G")
        LG_opt = LG

    H.arcsort(sort_type="olabel")
    LG_opt.arcsort(sort_type="ilabel")
    HLG = fst.compose(H, LG_opt)
    safe_connect(HLG, label="H o LG" if verbose_graph_ops else "")

    HLG.set_input_symbols(state_table)
    HLG.set_output_symbols(word_table)
    return HLG
