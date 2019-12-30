import numpy as np
rd = np.random
import itertools as iter

n_test_samples = 4
test_set_round = {(x,): np.array([np.round(x)]) for x in [rd.rand() for _ in range(n_test_samples)]}
n_training_samples = 1000
training_set_round = {(x,): np.array([np.round(x)]) for x in [rd.rand() for _ in range(n_test_samples)]}

n_test_samples = 10
test_set_cos = {(x,): np.array([np.cos(4*np.pi*x)/2+1/2]) for x in np.arange(0, 1, 1 / n_test_samples)}
n_training_samples = 10000
training_set_cos = {(x,): np.array([np.cos(4*np.pi*x)/2+1/2]) for x in np.arange(0, 1, 1 / n_training_samples)}

n_test_samples = 10
test_set_sin = {(x,): np.array([np.sin(4*np.pi*x)/2+1/2]) for x in np.arange(0, 1, 1 / n_test_samples)}
n_training_samples = 10000
training_set_sin = {(x,): np.array([np.sin(4*np.pi*x)/2+1/2]) for x in np.arange(0, 1, 1 / n_training_samples)}

test_set_or = {(i,j): np.array([i or j]) for i,j in iter.product([0,1], repeat=2)}
n_samples = 1000
training_set_or = {(x,y): np.array([np.round(x) or np.round(y)]) for (x,y) in [rd.rand(2) for _ in range(n_samples)]}

test_set_and = {(i,j): np.array([i and j]) for i,j in iter.product([0,1], repeat=2)}
n_samples = 1000
training_set_and = {(x,y): np.array([np.round(x) and np.round(y)]) for (x,y) in [rd.rand(2) for _ in range(n_samples)]}

n_samples = 10000
training_set_helix = {(x,): np.array([np.cos(4*np.pi*x)/2+1/2,np.sin(4*np.pi*x)/2+1/2]) for x in np.arange(0, 1 + 1 / n_training_samples, 1 / n_training_samples)}
n_test_samples = 10
test_set_helix = {(x,): np.array([np.cos(4*np.pi*x)/2+1/2,np.sin(4*np.pi*x)/2+1/2]) for x in np.arange(0, 1 + 1 / n_training_samples, 1 / n_test_samples)}

training_set_helix_special = {(np.float64(1),): np.array([1,0.5]), (np.float64(0.75),): np.array([0.5, 0])}
