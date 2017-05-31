import heapq as hq
import random

import numpy as np


def epsilon_greedy_policy(mdp, quality, eps):
    def policy(s):
        quals = [quality[(s, a)] for a in mdp.actions[s]]
        best_act = mdp.actions[s][np.argmax(quals)]
        rand_act = random.choice(mdp.actions[s])
        prob = random.uniform(0, 1)
        if prob < eps:
            return rand_act
        return best_act

    return policy


def deterministic_policy(mdp, quality):
    def policy(s):
        quals = [quality[(s, a)] for a in mdp.actions[s]]
        return mdp.actions[s][np.argmax(quals)]

    return policy


# TODO rewrite as for
def init_qual_model(mdp):
    quality = {(s, a): 0 for s in mdp.states for a in mdp.actions[s]}
    reward = {(s, a): 0 for s in mdp.states for a in mdp.actions[s]}
    transitions = {(s, a): random.choice(mdp.states) for s in mdp.states for a in mdp.actions[s]}
    model = {sa: (r, ss) for sa in reward for r in [reward[sa]] for ss in [transitions[sa]]}

    return quality, model


def dyna_q(mdp, alpha, gamma, n, eps, maxiter=None, theta=None):
    iterations = []
    cumrewards = []
    episodes = []

    quality, model = init_qual_model(mdp)

    cumreward = 0
    niter = 0
    epi_cnt = 0
    visited_states = {mdp.START_STATE}

    ss = mdp.START_STATE  # not sure
    while True:
        s = ss
        if s == mdp.END_STATE:
            epi_cnt += 1
            s = mdp.START_STATE

        greedy_policy = epsilon_greedy_policy(mdp, quality, eps)
        act = greedy_policy(s)
        r, ss = mdp.reward_transition(s, act)

        cumreward += r
        cumrewards.append(cumreward)
        iterations.append(niter)
        episodes.append(epi_cnt)

        visited_states.add(ss)
        diff_out = max([quality[(ss, a)] - quality[(s, act)] for a in mdp.actions[ss]])
        delta = diff_out
        quality[(s, act)] += alpha * (r + gamma * diff_out)
        model[(s, act)] = r, ss

        for _ in range(n):
            niter += 1

            ps = random.choice(list(visited_states))
            pa = random.choice(mdp.actions[ps])
            r, sss = model[(ps, pa)]
            diff_in = max([quality[(sss, a)] - quality[(ps, pa)] for a in mdp.actions[sss]])
            delta = max(delta, diff_in)
            quality[(ps, pa)] += alpha * (r + gamma * diff_in)

        niter += 1
        if theta is not None:
            if delta < theta:
                return deterministic_policy(mdp, quality), iterations, cumrewards, epi_cnt

        if maxiter is not None:
            if niter >= maxiter:
                return deterministic_policy(mdp, quality), iterations, cumrewards, epi_cnt


def dyna_q_prioritized_sweeping(mdp, alpha, gamma, n, eps, theta, maxiter):
    niter = 0
    cumreward = 0
    iterations = []
    cumrewards = []

    quality, model = init_qual_model(mdp)
    pqueue = []

    visited_states = {mdp.START_STATE}
    SS = mdp.START_STATE
    while True:
        S = SS
        greedy_policy = epsilon_greedy_policy(mdp, quality, eps)
        A = greedy_policy(S)
        R, SS = mdp.reward_transition(S, A)

        cumreward += R
        iterations.append(niter)
        cumrewards.append(cumreward)

        visited_states.add(SS)
        model[(S, A)] = R, SS
        max_diff = max(quality[(SS, a)] - quality[(S, A)] for a in mdp.actions[SS])
        P = abs(R + gamma * max_diff)
        if P > theta:
            hq.heappush(pqueue, (-P, S, A))
        for _ in range(n):
            if len(pqueue) == 0:
                break
            niter += 1
            _, s, act = hq.heappop(pqueue)
            r, ss = model[(s, act)]
            max_diff_in = max(quality[(ss, a)] - quality[(s, act)] for a in mdp.actions[ss])
            quality[(s, act)] += alpha * (r + gamma * max_diff_in)

            for ps in mdp.states:
                for pa in mdp.actions[ps]:
                    pr, ss = model[(ps, pa)]
                    if ss != s:
                        continue
                    max_diff_in_in = max(quality[(s, a)] - quality[(ps, pa)] for a in mdp.actions[s])
                    p = abs(pr + gamma * max_diff_in_in)
                    if p > theta:
                        hq.heappush(pqueue, (-p, ps, pa))

        niter += 1
        if maxiter is not None:
            if niter >= maxiter:
                return deterministic_policy(mdp, quality), iterations, cumrewards


def q_planning(mdp, alpha, gamma, maxiter):
    niter = 0
    cumreward = 0
    iterations = []
    cumrewards = []

    quality, _ = init_qual_model(mdp)
    while True:
        s = random.choice(mdp.states)
        act = random.choice(mdp.actions[s])
        r, ss = mdp.reward_transition(s, act)

        cumreward += r
        iterations.append(niter)
        cumrewards.append(cumreward)

        max_diff = max(quality[(ss, a)] - quality[(s, act)] for a in mdp.actions[ss])
        quality[(s, act)] += alpha * (r + gamma * max_diff)

        niter += 1

        if maxiter is not None:
            if niter >= maxiter:
                return deterministic_policy(mdp, quality), iterations, cumrewards
