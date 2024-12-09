"""
gen_filtered_hierarchical_data.py

This script contains functions for generating data following the procedure described in our paper "How transformers learn structured data: insights from hierarchical filtering."
It leverages numpy for numerical operations and multiprocessing for parallel generation of a large number of samples (N_trials).
The data is saved in a numpy file with the following format:
[q,l,sigma,x0s,xis,M,k] where:
- q: Size of the vocabulary
- l: Depth of the tree
- sigma: Standard deviation of the noise generating the grammar
- x0s: Array of root nodes of dim (N_trials)
- xis: Array of leaf nodes of dim (2**l,N_trials)
- M: Transition rates matrix of dim (q,q,q)
- k: Number of factorized layers

Functions:
- get_P_xlevel_root(M, level): Computes the probability matrices for a given level (i.e. Eq. (3) of our paper).
- get_leaves(x0, M, l, k, probs): Generates leaves from a root x0.
- get_M(q, sigma): Generates the transition rates matrix M.
- build_args(N_trials, M, l, q, k, probs): Builds the arguments for the parallel generation of samples.
- gen_tree(M, l, q, k, probs): Generates a couple root,leaves.

Dependencies:
- numpy
- scipy.special.softmax
- multiprocessing.Pool
- multiprocessing.cpu_count
"""

import numpy as np
from multiprocessing import Pool,cpu_count
from scipy.special import softmax

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

def get_leaves(x0,M,l,level,probs):
    def get_branches(S,M):
        # Get the two branches from a given state S using the standard rule
        p_flat = M[S,:,:].ravel()
        ind = np.arange(len(p_flat))
        return np.unravel_index(np.random.choice(ind, p=p_flat),M[S,:,:].shape)
    def get_nodes(x0,probs,level):
        nodes = np.empty(2**level,dtype=np.int8)
        q = probs.shape[0]
        for i in range(2**level):
            nodes[i] = np.random.choice(np.arange(q),p=probs[x0,:,i])
        return nodes
    x = get_nodes(x0,probs,level).astype(np.int8)
    for i in range(level+1,l+1):
        x_new = np.empty(2**i,dtype=np.int8)
        for j in range(len(x)):
            x_new[2*j],x_new[2*j+1] = get_branches(x[j],M)
        x = x_new
    return x

def get_M(q,sigma):
    h = np.log(np.zeros((q,q,q))) # Actually parametrize with logits
    M = np.empty((q,q,q))
    tuples = np.array([(i,j) for i in range(q) for j in range(q)]) # Generate the non-overlaping partitions
    np.random.shuffle(tuples)
    for i in range(q):
        for j in range(i*q,(i+1)*q):
            k,l = tuples[j]
            h[i,k,l] = sigma*np.random.randn()
        M[i,:,:] = softmax(h[i,:,:])
    return M

def build_args(N_trials,M,l,q,k,probs):
    arg_tuples = [(M,l,q,k,probs) for i in range(N_trials)]
    return arg_tuples

def gen_tree(M,l,q,k,probs):
    np.random.seed()
    x0 = np.random.randint(q) # Generate the root
    leaves = get_leaves(x0,M,l,k,probs)
    return x0,leaves

if __name__ == "__main__":
    p = Pool(cpu_count()) # Fix desired number of cpu threads for the generative process
    # Set the parameters for the generative process
    N_trials = int(1e6) # Number of samples to generate
    l = 4 # Depth of the tree
    q = 4 # Size of vocabulary
    sigma = 1.0 # Standard deviation of the noise generating the grammar
    seed = 0 # Seed for reproducibility
    k = 0 # Number of factorized layer, 0 <= k <= l
    x0s = np.empty(N_trials)
    xis = np.empty((int(2**l),N_trials),dtype=np.int8) # Save as integers to save memory
    M_s = np.empty((q,q,q)) # Transition rates
    # Generate the data
    np.random.seed(seed)
    M = get_M(q,sigma)
    probs = get_P_xlevel_root(M,k)
    runs = p.starmap(gen_tree,build_args(N_trials,M,l,q,k,probs)) # Will generate the data in parallel over p processes
    # Save the data in a convenient format
    for h in range(N_trials):
        x0s[h],xis[:,h] = runs[h]
    np.save('./data/labeled_data_factorized_{}_{}_{}_{}.npy'.format(q,l,sigma,k),np.array([q,l,sigma,x0s,xis,M,k],dtype=object))