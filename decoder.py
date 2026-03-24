import math
from collections import deque

import observation_model
import openfst_python as fst


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
        self._state_list = list(self.f.states())
        try:
            st = int(self.f.start())
            if st not in self._state_list:
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

    def initialise_decoding(self):
        self.V = []
        self.B = []
        self.W = []
        self.forward_computation_count = 0

        n = self._n_slots
        for t in range(self.om.observation_length() + 1):
            self.V.append([self.NLL_ZERO] * n)
            self.B.append([-1] * n)
            self.W.append([[] for _ in range(n)])

        st = int(self.f.start())
        self.V[0][st] = 0.0
        self.traverse_epsilon_active(0, [st])

    def traverse_epsilon_active(self, t, seed):
        q = deque(seed)
        seen = set(seed)
        while q:
            i = q.popleft()
            if self.V[t][i] == self.NLL_ZERO:
                continue
            for arc in self._eps_arcs[i]:
                j = int(arc.nextstate)
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
        prev_active = [i for i in self._state_list if self.V[t - 1][i] < self.NLL_ZERO]
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
        n = len(self.V[-1]) if self.V else 0
        for state in self.f.states():
            if state < 0 or state >= n:
                continue
            final_weight = float(self.f.final(state))
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
        while t <= self.om.observation_length():
            updated = self.forward_step(t)
            self.traverse_epsilon_active(t, updated)
            t += 1
        self.finalise_decoding()

    def backtrace(self):
        best_final_state = self.V[-1].index(min(self.V[-1]))
        best_state_sequence = [best_final_state]
        best_out_sequence = []

        t = self.om.observation_length()
        j = best_final_state

        while t >= 0:
            i = self.B[t][j]
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
