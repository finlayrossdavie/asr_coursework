#!/usr/bin/env python3

import os
import math
import numpy as np
import pexpect
from pexpect import TIMEOUT, EOF

kaldi_dir = None
nnetdir = "/group/teaching/asr/labs/tdnnf_mono_net/"
bindir = "/group/teaching/asr/labs/bin/"
if not kaldi_dir:
    path = ['/group/teaching/asr/labs/bin/lib/',
            '/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin/']
    path = ':'.join(path)
else:
    path = '{kd}/tools/openfst-1.6.5/lib/'.format(kd=kaldi_dir)
            
def initialize_nn(debug=False):
    nnet = pexpect.spawnu("{}/lab-chain-compute-post".format(bindir),
                          ["--feature-type=fbank",
                           "--frame-subsampling-factor=1",
                           "--frames-per-chunk=150",
                           "--cmvn-config={}/conf/cmvn.conf".format(nnetdir),
                           "--fbank-config={}/conf/fbank.conf".format(nnetdir),
                           "--global-cmvn-stats={}/conf/cmvn.gstat".format(nnetdir),
                           "{}/final.mdl".format(nnetdir),
                           "{}/den.fst".format(nnetdir),
                           "scp:-", "ark,t:-"], env={'LD_LIBRARY_PATH': path})
    try:
        nnet.expect("Ready.")
    except (TIMEOUT, EOF):
        nn_check_for_errors(debug)
    return nnet


def nn_check_for_errors(nnet, debug=False):
    if debug is True:
        import pdb
        pdb.set_trace()
    before = nnet.before if type(nnet.before) == str else ''
    after = nnet.after if type(nnet.after) == str else ''
    stdout = (before + '\r\n' + after).split('\r\n')
    err = [x for x in stdout if x.startswith('WARNING') or x.startswith('ERR')]
    raise Exception('\n'.join(err))


nnet = initialize_nn()
loaded_first_rec = False

class ObservationModel:
    def __init__(self, debug=False, kaldi_dir=None):
        self.debug = debug
        self.timesteps = 0
        self.dummy = False
        self.nnetdir = "/group/teaching/asr/labs/tdnnf_mono_net/"
        self.bindir = "/group/teaching/asr/labs/bin/"
        self.nnet = nnet
        self.state_map = self.load_state_map("{}/conf/pdfsmap".format(self.nnetdir))
        if not kaldi_dir:
            path = ['/group/teaching/asr/labs/bin/lib/',
                    '/opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin/']
            self.path = ':'.join(path)
        else:
            self.path = '{kd}/tools/openfst-1.6.5/lib/'.format(kd=kaldi_dir)

    def load_state_map(self, map_fn):
        state_map = {}
        with open(map_fn, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                state = '{}_{}'.format(line[2], int(line[5])+1)
                pdf = int(line[8])
                state_map[state] = pdf
        return state_map

    def load_audio(self, wav_fn):
        global loaded_first_rec
        if wav_fn[-3:] != 'wav':
            raise ValueError('Audio must be in wav format')
        if not (os.path.exists(wav_fn) and os.path.isfile(wav_fn)):
            raise FileNotFoundError("{} not found.".format(wav_fn))
        utt_name = wav_fn.split('.')[0] # has to be 16k

        if self.dummy:
            self.dummy = False

        if not self.nnet or not self.nnet.isalive():
            self.nnet = initialize_nn(self.debug)
        if loaded_first_rec:  # not first load data
            self.nnet.expect('LOG') # there's a previous line giving logprob over frames

        self.nnet.send("{} {}\n".format(utt_name, wav_fn))
        try:
            self.nnet.expect('LOG')
        except (TIMEOUT, EOF):
            nn_check_for_errors(self.nnet, self.debug)
        self.post_mat = self.parse_kaldi_post_mat(self.nnet.before)
        self.timesteps = len(self.post_mat)
        loaded_first_rec = True
        self.precompute_emission_nll()

    def parse_kaldi_post_mat(self, mat_str):
        mat_str = mat_str.split('\r\n')
        rows = []
        open_mat = False
        close_mat = False

        for line in mat_str:
            if not open_mat:
                idx = line.find('[')  # first line
                if idx != -1:
                    open_mat = True
            else:
                arr = line.strip().split()
                try:
                    if arr and arr[-1] != ']':
                        rows.append(np.array(arr, dtype='float32'))  # not last line
                    elif arr:
                        rows.append(np.array(arr[:-1], dtype='float32'))  # last line
                        close_mat = True
                except ValueError:
                    open_mat = False

        if not open_mat and close_mat:
            raise ValueError("Posterior matrix is malformed")
        if rows:
            mat = np.vstack(rows)
        else:
            nn_check_for_errors(self.nnet, self.debug)
            raise Exception("Something is wrong, matrix is empty.")
        open_mat = False
        close_mat = True
        return mat

    def load_dummy_audio(self):
        self.timesteps = 30
        self.dummy = True
        self.hmm_labels = []
        with open('phonelist.txt', 'r') as f:
            for ph in f.readlines():
                for i in range(1, 4):
                    self.hmm_labels.append("{}_{}".format(ph.strip(),i))
        self.precompute_emission_nll()

    def observation_length(self):
        return self.timesteps

    def precompute_emission_nll(self):
        """
        Precompute per-frame negative log-likelihoods for fast Viterbi (array indexing).
        Real audio: emission_nll[t-1, pdf_idx] = -log(post_mat[t-1, pdf_idx]).
        OOV HMM labels (e.g. #0_1): use oov_nll = log(n_pdf) per plan.
        Dummy: emission_nll[t-1, col] for a fixed label set (state_map + phonelist + #0_1).
        """
        T = self.timesteps
        if not self.dummy:
            pm = np.asarray(self.post_mat, dtype=np.float64)
            self.emission_nll = -np.log(np.maximum(pm, 1e-300))
            self.n_pdf = pm.shape[1]
            self.oov_nll = float(math.log(self.n_pdf))
            self._dummy_label_to_col = None
            return
        labels = sorted(set(self.state_map.keys()) | set(self.hmm_labels) | {'#0_1'})
        nlab = len(labels)
        self._dummy_label_to_col = {lab: j for j, lab in enumerate(labels)}
        self.emission_nll = np.zeros((T, nlab), dtype=np.float64)
        for ti in range(T):
            t = ti + 1
            for lab in labels:
                j = self._dummy_label_to_col[lab]
                lp = self.dummy_observation_probability(lab, t)
                self.emission_nll[ti, j] = -float(lp)
        self.n_pdf = nlab
        self.oov_nll = float(math.log(max(nlab, 1)))

    def pdf_idx_for_hmm_label(self, hmm_label):
        """Column index into emission_nll for this HMM string, or -1 if OOV (use oov_nll)."""
        hmm_label = '' if hmm_label is None else str(hmm_label)
        if not self.dummy:
            pdf_idx = self.state_map.get(hmm_label)
            return pdf_idx if pdf_idx is not None else -1
        if self._dummy_label_to_col is None:
            return -1
        j = self._dummy_label_to_col.get(hmm_label)
        return j if j is not None else -1

    def log_observation_probability(self, hmm_label, t):
        if t <= 0 or t > self.timesteps+1:
            raise IndexError("Timestep not in range [1,{}]".format(self.timesteps+1))
        # Normalise: SymbolTable may give non-str; missing ids -> '' from decoder helper.
        hmm_label = '' if hmm_label is None else str(hmm_label)

        if self.dummy:
            return self.dummy_observation_probability(hmm_label, t)

        try:
            pdf_idx = self.state_map.get(hmm_label)
            if pdf_idx is None:
                # Not in pdfsmap (e.g. #0_1) or unknown label: uniform over NN posteriors.
                n_pdf = self.post_mat.shape[1]
                return np.log(1.0 / n_pdf)
            v = self.post_mat[t - 1, pdf_idx]
            return np.log(float(v))
        except (KeyError, IndexError, TypeError):
            n_pdf = self.post_mat.shape[1]
            return np.log(1.0 / n_pdf)

    def dummy_observation_probability(self, hmm_label, t):
        """ Computes b_j(t) where j is the current state

        This is just a dummy version!  In later labs we'll generate
        probabilities for real speech frames.

        You don't need to look at this function in detail.

        Args: hmm_label (str): the HMM state label, j.  We'll use string form: "p_1", "p_2", "eh_1" etc
              t (int) : current time step, starting at 1

        Returns:
              p (float): the observation probability p(x_t | q_t = hmm_label)
        """

        hmm_label = '' if hmm_label is None else str(hmm_label)
        p = {}  # dictionary of probabilities

        assert(t > 0)

        # this is just a simulation!
        if t < 4:
            p = {'p_1': 1.0, 'p_2': 1.0, 'p_3': 1.0, 'eh_1': 0.2}
        elif t < 9:
            p = {'p_3': 0.5, 'eh_1': 1.0, 'eh_2': 1.0, 'eh_3': 1.0}
        elif t < 13:
            p = {'eh_3': 1.0, 'p_1': 1.0, 'p_2': 1.0, 'p_3': 1.0, 'er_1': 0.5}
        elif t < 18:
            p = {'p_3': 1.0, 'er_1': 1.0, 'er_2': 1.0, 'er_3': 0.7}
        elif t < 25:
            p = {'er_3': 1.0, 'z_1': 1.0, 'z_2': 1.0, 'z_3': 1.0}
        else:
            p = {'z_2': 0.5, 'z_3': 1.0}

        for label in self.hmm_labels:
            if label not in p:
                p[label] = 0.001  # give all other states a small probability to avoid zero probability

        # Disambiguation / any HMM label not listed above (e.g. #0_1 from WFST cascade)
        if hmm_label not in p:
            p[hmm_label] = 0.001

        # normalise the probabilities:
        scale = sum(p.values())
        for k in p:
            p[k] = p[k]/scale

        # Never KeyError: label may be OOV (e.g. #0_1) vs phonelist-based dummy.
        val = p.get(hmm_label)
        if val is None or val <= 0:
            val = 1.0 / max(len(p), 1)
        return np.log(val)
