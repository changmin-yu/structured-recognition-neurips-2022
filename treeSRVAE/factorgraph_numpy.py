'''
Implementation of a numpy-based version of factor graph and (loopy) belief propagation algorithm.
This implementation is adapted based on the py-factorgraph module (https://github.com/mbforbes/py-factorgraph)
Any error is mine and mine alone.
'''

import code  
import logging
import signal
import copy

import numpy as np

logger = logging.getLogger(__name__)

DEBUG_DEFAULT = True

LBP_MAX_ITERS = 50

E_STOP = False

def divide_safezero(a, b):
    c = a / b
    c[c == np.inf] = 0.0
    c = np.nan_to_num(c)
    return c

class Graph_numpy:
    def __init__(self, debug=DEBUG_DEFAULT):
        self.debug = debug

        self._rvs = {}
        self._factors = []

    def rv(self, name, n_opts, labels=[], meta={}, debug=DEBUG_DEFAULT):
        rv = RV(name, n_opts, labels, meta, debug)
        self.add_rv(rv)
        return rv

    def has_rv(self, rv_s):
        return rv_s in self._rvs

    def add_rv(self, rv):
        rv.meta['pruned'] = False
        if self.debug:
            assert rv.name not in self._rvs
        self._rvs[rv.name] = rv

    def get_rvs(self):
        return self._rvs

    def get_factors(self):
        return self._factors

    def remove_loner_rvs(self):
        removed = 0
        names = self._rvs.keys()
        for name in names:
            if self._rvs[name].n_edges() == 0:
                self._rvs[name].meta['pruned'] = True
                del self._rvs[name]
                removed += 1
        return removed

    def factor(self, rvs, name='', potential=None, meta={},
               debug=DEBUG_DEFAULT):
        for i in range(len(rvs)):
            if debug:
                assert isinstance(rvs[i], (str, RV))
            if isinstance(rvs[i], str):
                rvs[i] = self._rvs[rvs[i]]
            assert type(rvs[i]) is RV

        f = Factor(rvs, name, potential, meta, debug)
        self.add_factor(f)
        return f

    def add_factor(self, factor):
        if self.debug:
            assert factor not in self._factors
        self._factors += [factor]

    def joint(self, x):
        if self.debug:
            assert len(x) == len(self._rvs)

            for name, label in x.items():
                assert name in self._rvs
                assert self._rvs[name].has_label(label)

            for name, rv in self._rvs.items():
                assert name in x
                assert rv.has_label(x[name])

        prod = 1.0
        for f in self._factors:
            prod *= f.eval(x)
        return prod

    def bf_best_joint(self):
        return self._bf_bj_recurse({}, list(self._rvs.values()))

    def _bf_bj_recurse(self, assigned, todo):
        if len(todo) == 0:
            return assigned, self.joint(assigned)

        best_a, best_r = None, 0.0
        rv = todo[0]
        todo = todo[1:]
        for val in range(rv.n_opts):
            new_a = assigned.copy()
            new_a[rv.name] = val
            full_a, r = self._bf_bj_recurse(new_a, todo)
            if r > best_r:
                best_r = r
                best_a = full_a
        return best_a, best_r

    def lbp(self, init=True, normalize=False, max_iters=LBP_MAX_ITERS,
            progress=False):
        nodes = self._sorted_nodes()

        if init:
            self.init_messages(nodes)

        cur_iter, converged = 0, False
        while cur_iter < max_iters and not converged and not E_STOP:
            cur_iter += 1

            if progress:
                logger.debug('\titeration %d / %d (max)', cur_iter, max_iters)
                
            converged = True
            for n in nodes:
                n_converged = n.recompute_outgoing(normalize=normalize)
                converged = converged and n_converged

        return cur_iter, converged

    def _sorted_nodes(self):
        rvs = list(self._rvs.values())
        facs = self._factors
        nodes = rvs + facs
        return sorted(nodes, key=lambda x: x.n_edges())

    def init_messages(self, nodes=None):
        if nodes is None:
            nodes = self._sorted_nodes()
        for n in nodes:
            n.init_lbp()

    def print_sorted_nodes(self):
        print(self._sorted_nodes())

    def print_messages(self, nodes=None):
        if nodes is None:
            nodes = self._sorted_nodes()
        print('Current outgoing messages:')
        for n in nodes:
            n.print_messages()
            
    def get_pairs(self):
        return [f for f in self._factors if f.n_edges() == 2]
            
    def pairwise_marginals(self, pairs=None, normalize=False):
        if pairs is None:
            pairs = self.get_pairs()
        
        out_dict = {}
        for pair in pairs:
            name = pair.__repr__()
            margs = 1.0 * pair.get_potential()
            for i, rv in enumerate(pair._rvs):
                if i == 0:
                    margs = margs * rv.get_outgoing_for(pair)[:, None]
                elif i == 1:
                    margs = margs * rv.get_outgoing_for(pair)[None, :]
                else:
                    raise NotImplementedError
            if normalize:
                margs = margs / (np.sum(margs) + 1e-10)
            out_dict[name] = margs
        return out_dict

    def rv_marginals(self, rvs=None, normalize=False):
        if rvs is None:
            rvs = self._rvs.values()

        tuples = []
        out_dict = {}
        for rv in rvs:
            # Compute marginal
            name = str(rv)
            marg, _ = rv.get_belief()
            if normalize:
                marg /= np.sum(marg)

            tuples += [(rv, marg)]
            out_dict[name] = marg
        return tuples, out_dict

    def print_rv_marginals(self, rvs=None, normalize=False):
        disp = 'Marginals for RVs'
        if normalize:
            disp += ' (normalized)'
        disp += ':'
        print(disp)

        tuples, _ = self.rv_marginals(rvs, normalize)

        for rv, marg in tuples:
            print(str(rv))
            vals = range(rv.n_opts)
            if len(rv.labels) > 0:
                vals = rv.labels
            for i in range(len(vals)):
                print('\t', vals[i], '\t', marg[i])

    def debug_stats(self):
        logger.debug('Graph stats:')
        logger.debug('\t%d RVs', len(self._rvs))
        logger.debug('\t%d factors', len(self._factors))


class RV:

    def __init__(self, name, n_opts, labels=[], meta={}, debug=DEBUG_DEFAULT):
        if debug:
            for l in labels:
                assert isinstance(l, str)

            assert len(labels) == 0 or len(labels) == n_opts

        self.name = name
        self.n_opts = n_opts
        self.labels = labels
        self.debug = debug
        self.meta = meta 

        self._factors = []
        self._outgoing = None

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def get_factors(self):
        return self._factors

    def get_outgoing(self):
        return self._outgoing[:]

    def init_lbp(self):
        self._outgoing = [np.ones(self.n_opts) for f in self._factors]

    def print_messages(self):
        for i, f in enumerate(self._factors):
            print('\t', self, '->', f, '\t', self._outgoing[i])

    def recompute_outgoing(self, normalize=False):
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        old_outgoing = self._outgoing[:]

        total, incoming = self.get_belief()

        convg = True
        for i in range(len(self._factors)):
            o = divide_safezero(total, incoming[i])
            if normalize:
                o = divide_safezero(o, sum(o))
            self._outgoing[i] = o
            convg = convg and \
                sum(np.isclose(old_outgoing[i], self._outgoing[i])) == \
                self.n_opts
        return convg

    def get_outgoing_for(self, f):
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        for i, fac in enumerate(self._factors):
            if f == fac:
                return self._outgoing[i]

    def get_belief(self):
        incoming = []
        total = np.ones(self.n_opts)
        for i, f in enumerate(self._factors):
            m = f.get_outgoing_for(self)
            if self.debug:
                assert m.shape == (self.n_opts,)
            incoming += [m]
            total *= m
        return (total, incoming)

    def n_edges(self):
        return len(self._factors)

    def has_label(self, label):
        if len(self.labels) == 0:
            if self.debug:
                assert type(label) is int
            return label < self.n_opts
        else:
            if self.debug:
                assert isinstance(label, (int, str))
            if isinstance(label, str):
                return label in self.labels
            return label < self.n_opts

    def get_int_label(self, label):
        if type(label) is int:
            return label
        # assume string otherwise
        return self.labels.index(label)

    def attach(self, factor):
        if self.debug:
            for f in self._factors:
                assert f != factor, ('Can\'t re-add factor %r to rv %r' %
                                     (factor, self))

        self._factors += [factor]


class Factor:
    def __init__(self, rvs, name='', potential=None, meta={},
                 debug=DEBUG_DEFAULT):
        self.name = name
        self.debug = debug
        self.meta = meta  
        
        self._rvs = []
        self._potential = None
        self._outgoing = None

        for rv in rvs:
            self.attach(rv)

        if potential is not None:
            self.set_potential(potential)

    def __repr__(self):
        name = 'f' if len(self.name) == 0 else self.name
        return name + '(' + ', '.join([str(rv) for rv in self._rvs]) + ')'

    def n_edges(self):
        return len(self._rvs)

    def get_potential(self):
        return self._potential

    def get_rvs(self):
        return self._rvs

    def init_lbp(self):
        self._outgoing = [np.ones(r.n_opts) for r in self._rvs]

    def get_outgoing(self):
        return self._outgoing[:]

    def get_outgoing_for(self, rv):
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        for i, r in enumerate(self._rvs):
            if r == rv:
                return self._outgoing[i]

    def recompute_outgoing(self, normalize=False):
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        old_outgoing = self._outgoing[:]

        incoming = []
        belief = self._potential.copy()
        for i, rv in enumerate(self._rvs):
            m = rv.get_outgoing_for(self)
            if self.debug:
                assert m.shape == (rv.n_opts,)
            proj = np.ones(len(belief.shape), int)
            proj[i] = -1
            m_proj = m.reshape(proj)
            incoming += [m_proj]
            belief *= m_proj
            
        convg = True
        all_idx = list(range(len(belief.shape)))
        for i, rv in enumerate(self._rvs):
            rv_belief = divide_safezero(belief, incoming[i])
            axes = tuple(all_idx[:i] + all_idx[i+1:])
            o = rv_belief.sum(axis=axes)
            if self.debug:
                assert self._outgoing[i].shape == (rv.n_opts, )
            if normalize:
                o = divide_safezero(o, sum(o))
            self._outgoing[i] = o
            convg = convg and \
                sum(np.isclose(old_outgoing[i], self._outgoing[i])) == \
                rv.n_opts

        return convg

    def print_messages(self):
        for i, rv in enumerate(self._rvs):
            print('\t', self, '->', rv, '\t', self._outgoing[i])

    def attach(self, rv):
        if self.debug:
            for r in self._rvs:
                assert r != rv, 'Can\'t re-add RV %r to factor %r' % (rv, self)

        rv.attach(self)

        self._rvs += [rv]

        self._potential = None

    def set_potential(self, p):
        if self.debug:
            got = len(p.shape)
            want = len(self._rvs)
            assert got == want, ('potential %r has %d dims but needs %d' %
                                 (p, got, want))

            for i, d in enumerate(p.shape):
                got = d
                want = self._rvs[i].n_opts
                assert got == want, (
                    'potential %r dim #%d has %d opts but rv has %d opts' %
                    (p, i+1, got, want))

        self._potential = p

    def eval(self, x):
        if self.debug:
            for rv in self._rvs:
                assert rv.name in x
                assert rv.has_label(x[rv.name])

        ret = self._potential
        for r in self._rvs:
            ret = ret[r.get_int_label(x[r.name])]

        if self.debug:
            assert type(ret) is not np.ndarray

        return ret
    
if __name__=='__main__':
    # sanity checks
    g = Graph_numpy(debug=False)
    g.rv('v_0', 2)
    g.rv('v_1', 2)
    g.rv('v_2', 2)
    g.factor(['v_0'], potential=np.array([0.7, 0.3]))
    g.factor(['v_1'], potential=np.array([0.4, 0.6]))
    g.factor(['v_2'], potential=np.array([0.2, 0.8]))
    g.factor(['v_0', 'v_1'], potential=np.array([[0.5, 0.5], [0.5, 0.5]]))
    g.factor(['v_0', 'v_2'], potential=np.array([[0.5, 0.5], [0.5, 0.5]]))
    iters, cvged = g.lbp(normalize=True)
    g.print_messages()
    g.print_rv_marginals(normalize=True)