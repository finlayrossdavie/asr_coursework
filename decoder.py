import observation_model
import math
import openfst_python as fst

class MyViterbiDecoder:
    
    NLL_ZERO = 1e10
    
    def __init__(self, f, audio_file_name):
        self.om = observation_model.ObservationModel()
        self.f = f
        
        if audio_file_name:
            self.om.load_audio(audio_file_name)
        else:
            self.om.load_dummy_audio()
        
        self.initialise_decoding()

        
    def initialise_decoding(self):
        self.V = []
        self.B = []
        self.W = []
        self.forward_computation_count = 0  # ADDED
        
        for t in range(self.om.observation_length()+1):
            self.V.append([self.NLL_ZERO]*self.f.num_states())
            self.B.append([-1]*self.f.num_states())
            self.W.append([[] for i in range(self.f.num_states())])
        
        self.V[0][self.f.start()] = 0.0
        self.traverse_epsilon_arcs(0)        
        
    def traverse_epsilon_arcs(self, t):
        states_to_traverse = list(self.f.states())
        while states_to_traverse:
            
            i = states_to_traverse.pop(0)   
        
            if self.V[t][i] == self.NLL_ZERO:
                    continue
        
            for arc in self.f.arcs(i):
                
                if arc.ilabel == 0:
                  
                    j = arc.nextstate
                
                    if self.V[t][j] > self.V[t][i] + float(arc.weight):
                        
                        self.V[t][j] = self.V[t][i] + float(arc.weight)
                        self.B[t][j] = self.B[t][i] 
                        
                        if arc.olabel != 0:
                            self.W[t][j] = [arc.olabel]  # FIXED (was self.W[t][i] + [arc.olabel])
                        else:
                            self.W[t][j] = self.W[t][i]
                        
                        if j not in states_to_traverse:
                            states_to_traverse.append(j)

    
    def forward_step(self, t):
          
        for i in self.f.states():
            
            if not self.V[t-1][i] == self.NLL_ZERO:
                
                for arc in self.f.arcs(i):
                    
                    if arc.ilabel != 0:
                        self.forward_computation_count += 1  # ADDED
                        j = arc.nextstate
                        tp = float(arc.weight)
                        ep = -self.om.log_observation_probability(self.f.input_symbols().find(arc.ilabel), t)
                        prob = tp + ep + self.V[t-1][i]
                        if prob < self.V[t][j]:
                            self.V[t][j] = prob
                            self.B[t][j] = i
                            
                            if arc.olabel != 0:
                                self.W[t][j] = [arc.olabel]
                            else:
                                self.W[t][j] = []
                            
    
    def finalise_decoding(self):
        for state in self.f.states():
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
            self.forward_step(t)
            self.traverse_epsilon_arcs(t)
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

            if self.W[t][j]:  # FIXED - only collect when non-empty
                best_out_sequence = self.W[t][j] + best_out_sequence

            j = i  
            t -= 1

        best_state_sequence.reverse()
        
        # Returns a list of word strings  # FIXED - was a joined string
        word_sequence = [
            self.f.output_symbols().find(label)
            for label in best_out_sequence
            if label != 0
        ]
        
        return (best_state_sequence, word_sequence)