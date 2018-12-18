#  Copyright (c) 2018 Steven R. Brandt
#  Copyright (c) 2018 Christopher Taylor
#  Copyright (c) 2018 R. Tohid
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
#  Implementation ported from https://github.com/nlml/tsne_raw
#  Adjoining blog post: https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/
#
#  Special thanks to Liam Schoneveld for providing implementation and notes in
#  the public domain.
#
#import phylanx
#from phylanx.ast import *
from numpy import sum, power, log2, transpose, shape, reshape, vstack, dot, zeros, float64, exp, fill_diagonal, int64, expand_dims
from numpy.random import random, normal

#@Phylanx
def neg_squared_euc_distance(X):
    sum_X = sum(power(X, 2.0), axis=1)
    D = ((transpose((-2.0 * dot(X, transpose(X))) + sum_X)) + sum_X)
    return -D


#@Phylanx
def softmax(X, diag_zero=True):
    e_x = exp(X - reshape(X.max(axis=1), [-1, 1]))
    if diag_zero:
        fill_diagonal(e_x, 0.0)
    e_x = e_x + 1e-8

    return e_x / reshape(sum(e_x, axis=1), [-1, 1])


#@Phylanx
def calc_prob_mat(distances, sigmas=None):
    if sigmas is not None:
        two_sig_sq = 2.0 * power(reshape(sigmas, [-1, 1]), 2.0)
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)


#@Phylanx
def bin_search(eval_fn,
               dists,
               target,
               i_iter,
               sigma_,
               tol=1e-10,
               max_iter=10000,
               lower=1e-20,
               upper=1000.0):

    guess = 0.0

    for i in range(max_iter):
        guess = (lower + upper) / 2.0
        val = eval_fn(dists, guess, i)
        if val > target:
            upper = guess
        else:
            lower = guess

        if abs(val - target) <= tol:
            break

    return guess


#@Phylanx
def calc_perplexity(prob_matrix):
    entropy = -sum(prob_matrix * log2(prob_matrix), axis=1)
    # perplexity = power(2.0, entropy)
    return entropy


#@Phylanx
def perplexity(distances, sigmas):
    return calc_perplexity(calc_prob_mat(distances, sigmas))


#@Phylanx
def eval_fn(distances, sigma, i):
    return perplexity(distances[i:i + 1, :], sigma)


#@Phylanx
def find_optimal_sigmas(distances, target_perplexity):
    N = distances.shape[0]
    sigmas = zeros(N, dtype=float64)
    # TODO: parallelize this block?
    #
    for i in range(N):
        correct_sigma = bin_search(eval_fn, distances, target_perplexity, i, sigmas)
        sigmas[i] = correct_sigma
    return sigmas


#@Phylanx
def p_conditional_to_joint(P):
    return (P + transpose(P)) / (2.0 * float(P.shape[0]))


# NOTE: Ignore these for now...
#
# @Phylanx
# def q_joint(Y):
#     dists = neg_squared_euc_dists(Y)
#     exp_dists = exp(distances)
#     # TODO: does fill return a ref to inv_distances?
#     #
#     fill_diagonal(exp_dists, 0.0)
#
#     # TODO: return tuples?
#     #
#     return exp_dists / sum(exp_dists), None
#
# @Phylanx
# def symmetric_sne_grad(P, Q, Y):
#     pq_diff = P-Q
#     pq_expanded = expand_dims(pq_diff, 2) # NxNx1
#     y_diffs = expand_dims(Y, 1) - expand_dims(Y, 0) # NxNx2
#     grad = 4.0 * sum(pq_expanded * y_diffs, axis=1) # Nx2
#     return grad


#@Phylanx
def q_tsne(Y):
    distances = neg_squared_euc_distance(Y)
    inv_distances = power(1.0 - distances, -1.0)
    # TODO: does fill return a ref to inv_distnaces?
    #
    fill_diagonal(inv_distances, 0.0)

    # TODO: return tuples?
    #
    return inv_distances / sum(inv_distances), inv_distances


#@Phylanx
def tsne_grad(P, Q, Y, distances):
    pq_diff = P - Q
    pq_expanded = expand_dims(pq_diff, 2)  # NxNx1
    y_diffs = expand_dims(Y, 1) - expand_dims(Y, 0)  # NxNx2
    distances_expanded = expand_dims(distances, 2)  # NxNx1
    y_diffs_wt = y_diffs * distances_expanded  # NxNx2
    grad = 4.0 * sum(pq_expanded * y_diffs_wt, axis=1)  # Nx2
    return grad


#@Phylanx
def p_joint(X, target_perplexity):
    distances = neg_squared_euc_distance(X)
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    p_conditional = calc_prob_mat(distances, sigmas)
    P = p_conditional_to_joint(p_conditional)
    return P


#@Phylanx
def estimate_sne(X, y, P, rng, num_iters, q_fn, grad_fn, learning_rate,
                 momentum):
    shape_x_arr = zeros(2, dtype=int64)
    shape_x_arr[0] = X.shape[0]
    shape_x_arr[1] = 2

    # TODO:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html
    #Y = random('normal', (0.0, 0.0001, shape_x_arr))
    Y = normal(0.0, 0.0001, shape_x_arr)

    # TODO: original line didn't have shape_x_arr...originally expressed as...
    # Y = rng.normal(0.0, 0.0001, [shape(X, 0), 2])
    #

    if momentum:  # noqa: E999
        Y_m2 = Y.copy()  # TODO: copy
        Y_m1 = Y.copy()  # TODO: copy

    for i in range(num_iters):
        Q, distances = q_fn(Y)
        grads = grad_fn(P, Q, Y, distances)
        Y = Y - learning_rate * grads
        if momentum:
            Y += momentum * (Y_m1 - Y_m2)
            Y_m2 = Y_m1  # TODO: copy
            Y_m1 = Y  # TODO: copy

    return Y


if __name__ == "__main__":
    from numpy.random import rand, randint, RandomState

    SEED = 1
    PERPLEX = 20
    NUM_ITERS = 500
    LR = 10.0
    M = 0.9

    rstate = RandomState(SEED)
    nfeatures = 3
    samples = 10
    labels = 2

    X = rand(samples, nfeatures)
    y = randint(labels, size=samples)

    P = p_joint(X, PERPLEX)
    Y = estimate_sne(
        X,
        y,
        P,
        rstate,
        num_iters=NUM_ITERS,
        q_fn=q_tsne,
        grad_fn=tsne_grad,
        learning_rate=LR,
        momentum=M)

    print(Y)
