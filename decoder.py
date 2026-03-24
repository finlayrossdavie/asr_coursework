import math
from collections import deque

import observation_model
import openfst_python as fst


def count_fst_arcs(fst_obj):
    """Count arcs without using states() iterator (some bindings yield a trailing -1 sentinel)."""
    n = 0
    ns = int(fst_obj.num_states())
    for s in range(ns):
        try:
            n += sum(1 for _ in fst_obj.arcs(s))
        except Exception:
            continue
    return n


class MyViterbiDecoder:

    NLL_ZERO = 1e10

    def __init__(self, f, audio_file_name, beam=float("inf"), max_states=None, om=None):
        self.om = om if om is not None else observation_model.ObservationModel()
        self.f = f
        self.beam = beam
        self.max_states = max_states

        if audio_file_name:
            self.om.load_audio(audio_file_name)
        else:
            self.om.load_dummy_audio()

        self._ilabel_to_pdf = {}
        isym = self.f.input_symbols()
        for hmm_label, pdf_idx in self.om.state_map.items():
            try:
                il = isym.find(hmm_label)
            except Exception:
                continue
            if il is not None and il != -1:
                self._ilabel_to_pdf[il] = pdf_idx

        self._build_graph_cache()
        self.initialise_decoding()

    def _build_graph_cache(self):
        """Cache epsilon/emit arcs; set self._n_slots to fit every source id and arc destination."""
        # states() may yield a trailing -1 sentinel on some openfst_python builds
        self._state_list = []
        for s in self.f.states():
            try:
                si = int(s)
            except Exception:
                continue
            if si >= 0:
                self._state_list.append(si)
        try:
            st = int(self.f.start())
            if st >= 0 and st not in self._state_list:
                self._state_list.append(st)
        except Exception:
            pass

        ns = int(self.f.num_states())
        source_states = set(range(ns))
        source_states.update(self._state_list)
        try:
            source_states.add(int(self.f.start()))
        except Exception:
            pass

        hi = -1
        for i in source_states:
            try:
                hi = max(hi, int(i))
            except Exception:
                continue
        try:
            hi = max(hi, int(self.f.start()))
        except Exception:
            pass

        for i in source_states:
            try:
                ii = int(i)
                for arc in self.f.arcs(ii):
                    hi = max(hi, ii, int(arc.nextstate))
            except Exception:
                continue

        n = max(hi + 1, ns, 1)
        self._n_slots = n

        self._eps_arcs = [[] for _ in range(n)]
        self._emit_arcs = [[] for _ in range(n)]
        for i in sorted(source_states):
            if i < 0 or i >= n:
                continue
            ii = int(i)
            try:
                for arc in self.f.arcs(ii):
                    tp = float(arc.weight)
                    j = int(arc.nextstate)
                    if arc.ilabel == 0:
                        self._eps_arcs[ii].append(arc)
                    else:
                        pdf_idx = self._ilabel_to_pdf.get(arc.ilabel, -1)
                        self._emit_arcs[ii].append(
                            (j, tp, arc.ilabel, pdf_idx, arc.olabel)
                        )
            except Exception:
                continue

        self._ensure_n_slots_cover_cache_and_start()

    def _ensure_n_slots_cover_cache_and_start(self):
        """Extend DP / arc slot arrays so start and every cached nextstate index fits."""
        need = self._n_slots
        try:
            need = max(need, int(self.f.start()) + 1)
        except Exception:
            pass
        for row in self._eps_arcs:
            for arc in row:
                need = max(need, int(arc.nextstate) + 1)
        for row in self._emit_arcs:
            for tup in row:
                need = max(need, int(tup[0]) + 1)
        if need > self._n_slots:
            extra = need - self._n_slots
            self._eps_arcs.extend([[] for _ in range(extra)])
            self._emit_arcs.extend([[] for _ in range(extra)])
            self._n_slots = need

    def initialise_decoding(self):
        self.V = []
        self.B = []
        self.W = []
        self.forward_computation_count = 0

        n = self._n_slots
        try:
            need_start = int(self.f.start()) + 1
            if need_start > n:
                self._pad_n_slots(need_start)
                n = self._n_slots
        except Exception:
            pass

        T = self.om.observation_length()
        if T < 0:
            T = 0
        for t in range(T + 1):
            self.V.append([self.NLL_ZERO] * n)
            self.B.append([-1] * n)
            self.W.append([[] for _ in range(n)])

        if not self.V:
            return

        st = int(self.f.start())
        if st >= len(self.V[0]):
            self._pad_n_slots(st + 1)
            n = self._n_slots
            self.V = [[self.NLL_ZERO] * n for _ in self.V]
            self.B = [[-1] * n for _ in self.B]
            self.W = [[[] for _ in range(n)] for _ in self.W]

        st = int(self.f.start())
        self.V[0][st] = 0.0
        self.traverse_epsilon_active(0, [st])

    def _pad_n_slots(self, need):
        if need <= self._n_slots:
            return
        extra = need - self._n_slots
        self._eps_arcs.extend([[] for _ in range(extra)])
        self._emit_arcs.extend([[] for _ in range(extra)])
        self._n_slots = need

    def traverse_epsilon_active(self, t, seed):
        q = deque(seed)
        seen = set(seed)
        while q:
            i = q.popleft()
            if i < 0 or i >= len(self._eps_arcs) or i >= len(self.V[t]):
                continue
            if self.V[t][i] == self.NLL_ZERO:
                continue
            for arc in self._eps_arcs[i]:
                j = int(arc.nextstate)
                if j < 0 or j >= len(self.V[t]):
                    continue
                nw = self.V[t][i] + float(arc.weight)
                if nw < self.V[t][j]:
                    self.V[t][j] = nw
                    self.B[t][j] = self.B[t][i]
                    if arc.olabel != 0:
                        self.W[t][j] = [arc.olabel]
                    else:
                        self.W[t][j] = self.W[t][i]
                    if j not in seen:
                        seen.add(j)
                        q.append(j)

    def forward_step(self, t):
        prev_active = [
            i
            for i in self._state_list
            if 0 <= i < len(self.V[t - 1]) and self.V[t - 1][i] < self.NLL_ZERO
        ]
        if not prev_active:
            return []

        best_prev = min(self.V[t - 1][i] for i in prev_active)

        if self.max_states is not None:
            active_states = sorted(
                [(i, self.V[t - 1][i]) for i in prev_active], key=lambda x: x[1]
            )
            allowed = set(s[0] for s in active_states[: self.max_states])
        else:
            allowed = None

        updated = []
        for i in prev_active:
            if self.V[t - 1][i] > best_prev + self.beam:
                continue
            if allowed is not None and i not in allowed:
                continue

            if i < 0 or i >= len(self._emit_arcs):
                continue
            for j, tp, _il, pdf_idx, olabel in self._emit_arcs[i]:
                self.forward_computation_count += 1
                if (
                    pdf_idx < 0
                    or self.om.emission_nll is None
                    or pdf_idx >= self.om.emission_nll.shape[1]
                ):
                    ep = self.NLL_ZERO
                else:
                    ep = self.om.emission_nll[t - 1, pdf_idx]
                if j < 0 or j >= len(self.V[t]):
                    continue
                prob = tp + ep + self.V[t - 1][i]
                if prob < self.V[t][j]:
                    self.V[t][j] = prob
                    self.B[t][j] = i
                    if olabel != 0:
                        self.W[t][j] = [olabel]
                    else:
                        self.W[t][j] = []
                    updated.append(j)
        return updated

    def finalise_decoding(self):
        if not self.V:
            return
        n = len(self.V[-1])
        # Avoid f.states(): openfst_python may include a -1 sentinel -> "State ID -1 not valid"
        for state in range(n):
            try:
                final_weight = float(self.f.final(state))
            except Exception:
                continue
            if self.V[-1][state] != self.NLL_ZERO:
                if final_weight == math.inf:
                    self.V[-1][state] = self.NLL_ZERO
                else:
                    self.V[-1][state] += final_weight

        finished = [x for x in self.V[-1] if x < self.NLL_ZERO]
        if not finished:
            print("No path got to the end of the observations.")

    def decode(self):
        self.initialise_decoding()
        t = 1
        T = max(0, self.om.observation_length())
        while t <= T:
            updated = self.forward_step(t)
            self.traverse_epsilon_active(t, updated)
            t += 1
        self.finalise_decoding()

    def backtrace(self):
        n = len(self.V[-1]) if self.V else 0
        finite = [
            (s, self.V[-1][s])
            for s in range(n)
            if self.V[-1][s] < self.NLL_ZERO
        ]
        if not finite:
            return ([], [])

        best_final_state = min(finite, key=lambda x: x[1])[0]
        best_state_sequence = [best_final_state]
        best_out_sequence = []

        t = min(self.om.observation_length(), len(self.V) - 1)
        j = best_final_state

        while t >= 0:
            i = self.B[t][j]
            if i < 0:
                break
            best_state_sequence.append(i)

            if self.W[t][j]:
                best_out_sequence = self.W[t][j] + best_out_sequence

            j = i
            t -= 1

        best_state_sequence.reverse()

        word_sequence = [
            self.f.output_symbols().find(label)
            for label in best_out_sequence
            if label != 0
        ]

        return (best_state_sequence, word_sequence)
