"""
Functions to implement belief propagation on a tree using the fast Python library Numba. 
The user should use the function run_BP to perform the inference (either of the root or of masked symbols)
"""

import numpy as np
from numba import njit
from multiprocessing import Pool,cpu_count
from time import time

@njit
def generate_tree(l,q,leaves):
    up_messages = np.zeros((l+1,2**l,q))
    down_messages = np.zeros((l+1,2**l,q))
    for i in range(l):
        for j in range(2**i):
            up_messages[i,j,:] = 1/q
            down_messages[i,j,:] = 1/q
    up_messages[l,:,:] = 1/q
    for j in range(2**l):
        down_messages[l,j,:] += leaves[j,:] # Add the prescribed leaves
    return up_messages, down_messages

def update_messages(l,q,up_messages,down_messages,M,factorized_layers=0):
    def get_P_xlevel_root(M,level):
        M_L = np.sum(M,axis=2)
        M_R = np.sum(M,axis=1)
        q = M_L.shape[0]
        leaves_indices = np.arange(2**(level+1))[2**level:]
        leaves_indices_binary = [bin(leaves_indices[i])[2:][1:] for i in range(len(leaves_indices))]
        probs = np.empty((q,q,2**level))
        probs_prev = np.empty((q,q,2**level))
        for i in range(2**level):
            probs[:,:,i] = np.eye(q)
            probs_prev[:,:,i] = np.eye(q)
            for j in range(level):
                if leaves_indices_binary[i][j] == '0':
                    probs[:,:,i] = probs_prev[:,:,i]@M_L
                else:
                    probs[:,:,i] = probs_prev[:,:,i]@M_R
                probs_prev[:,:,:] = probs[:,:,:]
        return probs 
    # Pre allocate stuff
    r_up = np.zeros(q)
    l_up = np.zeros(q)
    v_down = np.zeros(q)
    # Start from the leaves and go up to update downgoing (root to leaves) messages
    #for i in range(l-1,-1,-1):
    for i in range(l-1,factorized_layers-1,-1):
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            # Update the outgoing messages
            v_down[:] = 0
            for p1 in range(q): # Not using @ because M matrix is not contiguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        v_down[p1] += l_down[p2]*M[p1,p2,p3]*r_down[p3]
            down_messages[i,j,:] = v_down/np.sum(v_down)
    if factorized_layers > 0:
        probs = get_P_xlevel_root(M,factorized_layers)
        down_messages_factorized = np.zeros((3,2**factorized_layers,q)) # Along axis = 1 have before and after the factor node and then the variable node
        down_messages_factorized[-1,:2**factorized_layers,:] = down_messages[factorized_layers,:2**factorized_layers,:]
        for j in range(2**factorized_layers): # Do each of the 2**k factor nodes updates
            for p1 in range(q):
                for p2 in range(q):
                    down_messages_factorized[1,j,p1] += probs[p1,p2,j]*down_messages_factorized[-1,j,p2]
            down_messages_factorized[1,j,:] = down_messages_factorized[1,j,:]/np.sum(down_messages_factorized[1,j,:])
        down_messages_factorized[0,0,:] = np.prod(down_messages_factorized[1,:,:],axis=0)/np.sum(np.prod(down_messages_factorized[1,:,:],axis=0))
        down_messages[0,0,:] = down_messages_factorized[0,0,:]
        # Now go back down
        up_messages_factorized = np.zeros((3,2**factorized_layers,q))
        up_messages_factorized[0,0,:] = 1/q
        for j in range(2**factorized_layers):
            mask = np.arange(2**factorized_layers) != j
            up_messages_factorized[1,j,:] = np.prod(down_messages_factorized[1,mask,:],axis=0)*up_messages_factorized[0,0,:]
            up_messages_factorized[1,j,:] = up_messages_factorized[1,j,:]/np.sum(up_messages_factorized[1,j,:])
            for p1 in range(q):
                for p2 in range(q):
                    up_messages_factorized[-1,j,p1] += probs[p2,p1,j]*up_messages_factorized[1,j,p2]
            up_messages_factorized[-1,j,:] = up_messages_factorized[-1,j,:]/np.sum(up_messages_factorized[-1,j,:])
        up_messages[factorized_layers,:2**factorized_layers,:] = up_messages_factorized[-1,:,:]
    for i in range(factorized_layers,l):
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            v_up = up_messages[i,j,:]
            # Update the outgoing messages
            r_up[:] = 0
            l_up[:] = 0
            v_down[:] = 0
            for p1 in range(q): # Not using @ because M matrix is not conitguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        r_up[p1] += v_up[p2]*M[p2,p3,p1]*l_down[p3]
                        l_up[p1] += v_up[p2]*M[p2,p1,p3]*r_down[p3]
            up_messages[i+1,2*j,:] = l_up/np.sum(l_up)
            up_messages[i+1,2*j+1,:] = r_up/np.sum(r_up)
    return up_messages,down_messages

@njit
def compute_marginals(l,q,up_messages,down_messages):
    marginals = np.empty((l+1,2**l,q))
    for i in range(l+1):
        for j in range(2**i):
            marginals[i,j,:] = up_messages[i,j,:]*down_messages[i,j,:]
            marginals[i,j,:] = marginals[i,j,:]/np.sum(marginals[i,j,:])
    return marginals

@njit
def get_freeEntropy(M,l,q,up_messages,down_messages,factorized_layers=0):
    if factorized_layers > 0:
        M_L = np.sum(M,axis=2)
        M_R = np.sum(M,axis=1)
    # First compute the free entropy from the variables
    F_variables = 0
    for i in range(1,l): # Exclude both the root and the leaves
        for j in range(2**i):
            F_variables += np.log(np.sum(up_messages[i,j,:]*down_messages[i,j,:]))/np.log(q)
    # Now compute the free entropy from the factors
    F_factors = 0
    for i in range(l):
        if i < factorized_layers:
            M_eff = np.empty((q,q,q))
            for j in range(q):
                M_eff[j,:,:] = np.outer(M_L[j,:],M_R[j,:])
        else:
            M_eff = M
        for j in range(2**i):
            l_down = down_messages[i+1,2*j,:]
            r_down = down_messages[i+1,2*j+1,:]
            v_up = up_messages[i,j,:]
            z_factor = 0
            for p1 in range(q): # Not using @ because M matrix is not conitguous so better performance this way
                for p2 in range(q):
                    for p3 in range(q):
                        z_factor += v_up[p1]*M_eff[p1,p2,p3]*l_down[p2]*r_down[p3]
            F_factors += np.log(z_factor)/np.log(q)
    return -(F_factors - F_variables)/2**l
    
def run_BP(M,l,q,xis,factorized_layers=0):
    # Convert the leaves into messages, not super efficient but sequences are not so long and just need to do it once
    leaves_BP = np.empty((len(xis),q))
    for i in range(len(xis)):
        if xis[i] == q + 1: # Masked symbols
            leaves_BP[i,:] = 1/q
        else:
            leaves_BP[i,:] = 0
            leaves_BP[i,xis[i]] = 1
    up_messages,down_messages = generate_tree(l,q,leaves_BP)
    up_messages,down_messages = update_messages(l,q,up_messages,down_messages,M,factorized_layers)
    freeEntropy = get_freeEntropy(M,l,q,up_messages,down_messages,factorized_layers)
    marginals = compute_marginals(l,q,up_messages,down_messages)
    return marginals,freeEntropy

def masked_inference(M,l,q,xis,factorized_layers=0):
    marginals,_ = run_BP(M,l,q,xis,factorized_layers)
    return marginals[-1,:,:]

def root_inference(M,l,q,xis,factorized_layers=0):
    marginals,_ = run_BP(M,l,q,xis,factorized_layers)
    return marginals[0,:,:]

def run_root_inference(M,l,q,xis,x0s,N_trials,factorized_layers=0):
    p = Pool(cpu_count())
    runs = p.starmap(root_inference,[(M,l,q,xis[:,k],factorized_layers) for k in range(int(N_trials))])
    success = np.empty((N_trials))
    for i in range(N_trials):
        success[i] = np.argmax((runs[i])[0,:]) == x0s[i]
    success_rates = np.mean(success)
    p.close()
    return success_rates

def run_MLM_inference(M,l,q,xi,mask_rate,factorized_layers=0):
    np.random.seed()
    xi_masked = np.copy(xi)
    masked_indices = np.random.choice(len(xi),size=int(mask_rate*len(xi)),replace=False)
    xi_masked[masked_indices] = q + 1
    marginals = masked_inference(M,l,q,xi_masked,factorized_layers)
    success_rate = np.mean(np.argmax(marginals[masked_indices,:],axis=1) == xi[masked_indices])
    return success_rate

def MLM_BP_accuracy(M,l,q,xi,mask_rate,N_trials,factorized_layers=0):
    p = Pool(cpu_count())
    runs = p.starmap(run_MLM_inference,[(M,l,q,xi[:,k],mask_rate,factorized_layers) for k in range(int(N_trials))])
    success_rates = np.mean(runs)
    p.close()
    return success_rates