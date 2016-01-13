# -*- coding: utf-8 -*-


from collections import deque

import numpy as np


class AlternateTree:
    def __init__(self, bipartite_graph, root):
        self.bg = bipartite_graph
        self.S = [False for _ in range(self.bg.n)]  # Only stores belonging:
        # fathers are given by bg matching
        self.T = [False for _ in range(self.bg.n)]
        self.father = [None for _ in range(self.bg.n)]  # Stores fathers
        self.stack = deque()
        # append, popleft to use it as a FIFO => breadth first search
        self.explore(root)

    def explore(self, u):
        # Only for u in first part of the graph
        if not self.S[u]:  # Do nothing if already explored
            # Mark as explore
            self.S[u] = True
            # Get neighbours
            neighbours = self.bg.neighbours_in_G_l(u)
            for n in neighbours:
                if self.father[n] is None:
                    self.father[n] = u  # Track u as father
                    self.stack.append(n)

    def find_augmenting_path(self):
        found = None
        while found is None:  # Stopped by return when path is found
            try:
                v = self.stack.popleft()
                self.T[v] = True
                u = self.bg.matching[1][v]
                if u is None:
                    found = v
                else:
                    self.explore(u)
            except IndexError:
                # Augment labelling
                slacks = self.bg.slack_matrix()
                # Get minimal slack
                s = [u for (u, is_in) in enumerate(self.S) if is_in]
                t = [v for (v, is_in) in enumerate(self.T) if not is_in]
                alpha = np.min(slacks[s, :][:, t])
                argmins = np.nonzero(slacks == alpha)  # As two separate lists
                # of indices
                argmins = zip(argmins[0], argmins[1])  # As one list of couples
                # Update labels
                for i in range(self.bg.n):
                    if self.S[i]:
                        self.bg.labels[0][i] -= alpha
                    if self.T[i]:
                        self.bg.labels[1][i] += alpha
                # Update stack for new neighbours
                for (u, v) in argmins:
                    if self.S[u] and self.father[v] is None:
                        self.father[v] = u
                        self.stack.append(v)
        return found


class BipartiteGraph:

    def __init__(self, weights):
        self.w = weights
        self.n, _ = weights.shape  # Cardinal of both parts of the graph
        self.labels = [None, None]  # Labels for both parts of the graph
        self.matching = [None, None]  # Matching for both parts of the graph

    def init_labels(self):
        self.labels[0] = [np.max(self.w[u, :]) for u in range(self.n)]
        self.labels[1] = [0 for _ in range(self.n)]

    def init_matching(self):
        # Init empty matching
        self.matching[0] = [None for _ in range(self.n)]
        self.matching[1] = [None for _ in range(self.n)]

    def slack(self, u, v):
        # u is from first part, v from second
        return self.labels[0][u] + self.labels[1][v] - self.w[u, v]

    def slack_matrix(self):
        return (np.array(self.labels[0])[:, np.newaxis] +
                np.array(self.labels[1])[np.newaxis, :] -
                self.w)

    def neighbours_in_G_l(self, u):
        # u belongs to first part of the graph
        return [v for v in range(self.n) if self.slack(u, v) == 0]

    def augment_matching(self, tree, v):
        # Augments matching from given tree and leaf
        while v is not None:
            u = tree.father[v]
            self.matching[1][v] = u
            next_v = self.matching[0][u]  # None if u is the root of the tree
            self.matching[0][u] = v
            v = next_v

    def max_weight_matching(self):
        self.init_labels()
        self.init_matching()
        u = 0  # Init free vertex from first part
        while u < self.n:
            if self.matching[0][u] is None:
                # Find augmenting path from u
                # Init alternate tree
                tree = AlternateTree(self, u)
                # Augment matching from alternate tree
                v = tree.find_augmenting_path()  # Might also augment labels
                self.augment_matching(tree, v)
            else:
                # Find an other u...
                u += 1
        # Matching is complete
        return self.matching[0]
        # TODO consider returning also reverse matching


def min_weight_perm(w):
    """Returns matchings s (as a list) such as sum of weights w[i, s[i]]
    is minimal.
    If w is square, s is a permutation, if first dimension of w is smaller
    than second, s is not surjective, else s in not defined everywhere and
    None is put on undifined values.
    Uses so-called Hugarian method based on Kuhn-Munkres theory.
    (see notes, here the problem is inversed: we search the minimum)
    """
    n1, n2 = w.shape
    if n1 < n2:
        m = np.max(w)
        w = np.vstack([w, (m + 1) * np.ones((n2 - n1, n2))])
    elif n1 > n2:
        return reverse_function(min_weight_perm(w.T), n1)
    return _min_weight_perm_square(w)[:n1]


def reverse_function(s, size=None):
    """Reverse funcion s, assuming that s goes from an integer intervalle
    to an other. if s is not surjective some values are populated with None.
    size is the size of the second intervalle. If not given max(s) is used.
    """
    if size is None:
        size = max(s)
    rev = [None for _ in range(size)]
    for i, j in enumerate(s):
        rev[j] = i
    return rev


def _min_weight_perm_square(w):
    m = np.max(w)
    bg = BipartiteGraph(m - w)
    return bg.max_weight_matching()


def permutation_matrix(perm):
    n = len(perm)
    perm_mat = np.zeros((n, n))
    perm_mat[range(n), perm] = 1
    return perm_mat


def weight(perm, w):
    return np.sum(np.multiply(w, permutation_matrix(perm)))


def min_weight(w):
    """Returns weight of the minimal permutation.
    """
    return weight(min_weight_perm(w), w)


def all_perms(l):
    if len(l) <= 1:
        yield l
    else:
        for perm in all_perms(l[1:]):
            for i in range(len(l)):
                yield list(perm[:i]) + [l[0]] + list(perm[i:])


def min_weight_perm_brute_force(w):
    n, _ = w.shape
    min_w = np.sum(w.diagonal())
    min_perm = range(n)
    for perm in all_perms(range(n)):
        cur_w = weight(perm, w)
        if cur_w < min_w:
            min_w = cur_w
            min_perm = perm
    return min_perm
