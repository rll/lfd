#!/usr/bin/env python

try:
    import gurobipy as grb
    USE_GUROBI = True
    GRB = grb.GRB # constants for gurobi
except ImportError:
    USE_GUROBI = False
import h5py
import sys
import os

import numpy as np

eps = 10**-8
MAX_ITER=1000

"""
functions  and classes for setting up and running max margin, uses gurobi as the basic optimizer
"""


class MaxMarginModel(object):
    
    def __init__(self, actions, N):
        """
        @param actions: list of the actions that we're going to be deciding between
        @param N: number of features for each state/action pair, controls the number of 
                  variables in the optimization
        """
        self.model = grb.Model()
        self.actions = actions[:]
        self.N = N
        self.w = np.asarray([self.model.addVar(lb = -1*GRB.INFINITY, name = str(i)) 
                             for i in range(N)])
        self.w0 = self.model.addVar(lb = -1*GRB.INFINITY, name='w0', obj=0)
        self.weights = np.zeros(N)
        self.xi = self.model.addVar(lb = 0, name = 'xi')
        self.xi_val = None
        self.model.update()
        # w'*w + C*xi
        self.model.setObjective(np.dot(self.w, self.w) + self.xi) # C parameter is set at optimization time
        self.model.update()
        self.model.setObjective(np.dot(self.w, self.w)) # we'll be adding in slacks per constraint
        self.model.remove(self.xi)
        self.xi = []            #  list to keep track of slack variables
        self.xi_val = []



    @staticmethod
    def read(fname, actions, num_features):
        mm_model = MaxMarginModel.__new__(MaxMarginModel)
        MaxMarginModel.read_helper(mm_model, fname, actions, num_features)
        assert len(mm_model.model.getVars()) == 1+ len(mm_model.xi) + len(mm_model.w), \
            "Number of Gurobi vars mismatches the MaxMarginModel vars"
        ## need to add 1 for constant offset
        return mm_model

    @staticmethod
    def read_helper(mm_model, fname, actions, num_features):
        mm_model.actions = actions[:]
        grb_model = grb.read(fname)
        mm_model.model = grb_model
        w = []
        for var in mm_model.model.getVars():
            try:
                if var.VarName == 'w0':
                    mm_model.w0 = var
                    continue
                int(var.VarName)
                w.append(var)
            except ValueError:
                pass
        mm_model.N = len(w)
        assert mm_model.N == num_features, "Number of features in the model mismatches the number of the feature vector"
        mm_model.w = np.asarray(w)
        mm_model.populate_slacks()
        mm_model.weights = np.zeros(len(w))

    def populate_slacks(self):
        self.xi = [var for var in self.model.getVars() if var.VarName.startswith('xi')]
        self.xi_val = []

    def add_constraint(self, expert_action_phi, rhs_action_phi, margin_value, xi_var, update=True):
        """
        function to add a constraint to the model with pre-computed
        features and margins
        """
        lhs_coeffs = [(p, w) for p, w in zip(expert_action_phi, self.w) if abs(p) >= eps]
        lhs_coeffs.append((1, self.w0))
        lhs = grb.LinExpr(lhs_coeffs)
        rhs_coeffs = [(p, w) for w, p in zip(self.w, rhs_action_phi) if abs(p) >= eps]
        rhs_coeffs.append((-1, xi_var))
        rhs_coeffs.append((1, self.w0))
        rhs = grb.LinExpr(rhs_coeffs)
        rhs += margin_value
        self.model.addConstr(lhs >= rhs)
        #store the constraint so we can store them to a file later
        if update:
            self.model.update()

    def add_xi(self, xi_name = None):
        if not xi_name:
            xi_name = 'xi_{}'.format(len(self.xi))
        new_xi = self.model.addVar(lb = 0, name = xi_name, obj = 1)
        self.xi.append(new_xi)
        self.model.update()
        return new_xi
        
    def load_constraints_from_file(self, fname, verbose=True, max_constrs = None):
        """
        loads the contraints from the file indicated and adds them to the optimization problem
        """
        infile = h5py.File(fname, 'r')
        n_other_keys = 0
        if 'weights' in infile:
            self.weights = infile['weights'][:]
            n_other_keys += 1
        if 'xi' in infile:
            self.xi_val = infile['xi'][:]
            n_other_keys += 1
        if 'w0' in infile:
            self.w0_val = infile['w0'][()]
            n_other_keys += 1
        slack_names = {}
        total_constrs = len(infile) - n_other_keys
        n_added = 0
        for key_i in infile:
            if n_added > max_constrs:
                break
            constr = infile[str(key_i)]
            if constr['exp_action_name'][()] not in self.actions:
                continue
            exp_phi = constr['exp_phi'][:]
            rhs_phi = constr['rhs_phi'][:]
            margin = constr['margin'][:]
            slack_var = self.add_xi()
            n_constrs = rhs_phi.shape[0]
            assert len(margin.shape) == 1 and len(margin) == n_constrs
            for i in range(n_constrs):
                self.add_constraint(exp_phi, rhs_phi[i], margin[i], slack_var, update=False)
            if verbose:
                sys.stdout.write("\rAdded Constraint Group {}/{}          ".format(key_i, total_constrs))
                sys.stdout.flush()
        infile.close()
        if verbose:
            print ""
        self.model.update()

    def save_weights_to_file(self, fname):
        # changed to use h5py.File so file i/o is consistent
        outfile = h5py.File(fname, 'a')
        if 'weights' in outfile:
            del outfile['weights']
        outfile['weights'] = self.weights
        if 'w0' in outfile:
            del outfile['w0']
        outfile['w0'] = self.w0_val
        if 'xi' in outfile:
            del outfile['xi']
        outfile['xi'] = self.xi_val
        outfile.close()


    def load_weights_from_file(self, fname):
        infile = h5py.File(fname, 'r')
        self.weights = infile['weights'][:]
        self.w0_val = infile['w0'][()]
        if 'xi' in infile:
            self.xi_val = infile['xi'][:]
        infile.close()
    
    def scale_objective(self, C):    
        self.model.update()
        for xi_var in self.xi:
            xi_var.Obj *= C
        self.model.update()

    def margin_err(self):
        return sum(xi.X for xi in self.xi)

    def optimize_model(self):
        self.model.update()
        self.model.optimize()
        try:
            self.weights = [x.X for x in self.w]
            self.w0_val = self.w0.X
            self.xi_val = [x.X for x in self.xi]
            return self.weights, self.w0_val
        except grb.GurobiError:
            raise RuntimeError, "issue with optimizing model, check gurobi optimizer output"

    def save_model(self, fname):
        self.model.write(fname)


class BellmanMaxMarginModel(MaxMarginModel):    
    
    def __init__(self, actions, N, gamma=1):
        MaxMarginModel.__init__(self, actions, N)
        self.action_reward = -1
        self.goal_reward = 10
        self.dead_end_value = -10
        self.yi = []
        self.yi_val = []
        self.gamma = gamma
        self.model.update()

    @staticmethod
    def read(fname, actions, num_features):
        mm_model = BellmanMaxMarginModel.__new__(BellmanMaxMarginModel)
        MaxMarginModel.read_helper(mm_model, fname, actions, num_features)
        assert len(mm_model.model.getVars()) == len(mm_model.xi) + len(mm_model.yi)+ len(mm_model.w) + 1, "Number of Gurobi vars mismatches the BellmanMaxMarginModel vars" # constant 1 is for w0
        param_fname = mm_model.get_param_fname(fname)
        param_f = h5py.File(param_fname, 'r')
        mm_model.action_reward = param_f['action_reward'][()]
        mm_model.goal_reward = 10
        mm_model.gamma = param_f['gamma'][()]
        return mm_model

    def bellman_err(self):
        return sum(yi.X for yi in self.yi)

    def populate_slacks(self):
        self.xi = [var for var in self.model.getVars() if var.VarName.startswith('xi')]
        self.xi_val = []
        self.yi = [var for var in self.model.getVars() if var.VarName.startswith('yi')]
        self.yi_val = []
    
    def add_bellman_constraint(self, curr_action_phi, next_action_phi, update=True, final_transition=False):
        lhs_coeffs = [(p, w) for w, p in zip(self.w, curr_action_phi) if abs(p) >= eps]
        lhs = grb.LinExpr(lhs_coeffs)
        if final_transition:
            rhs_coeffs = []
        else:
            rhs_coeffs = [(self.gamma*p, w) for w, p in zip(self.w, next_action_phi) if abs(p) >= eps]
        yi_pos_var, yi_neg_var = self.add_yi()
        rhs_coeffs.append((1, yi_pos_var)) 
        rhs_coeffs.append((-1, yi_neg_var)) 
        rhs = grb.LinExpr(rhs_coeffs)
        rhs += self.action_reward
        if final_transition:
            rhs += self.goal_reward
        # w'*curr_phi == -1 + yi_pos - yi_neg + gammma * w'*next_phi
        self.model.addConstr(lhs == rhs)
        #store the constraint so we can store them to a file later
        if update:
            self.model.update()

    def add_deadend_constraint(self, phi, yi_pos_var, update=True):
        lhs_coeffs = [(p, w) for w, p in zip(self.w, phi) if abs(p) >= eps]
        lhs = grb.LinExpr(lhs_coeffs)
        rhs_coeffs = [(1, yi_pos_var)]
        rhs = grb.LinExpr(rhs_coeffs)
        rhs += self.dead_end_value
        self.model.addConstr(lhs <= rhs)
        if update:
            self.model.update()

    def add_yi(self, pos_only=False):
        yi_pos_name = 'yi_pos_{}'.format(len(self.yi))
        yi_neg_name = 'yi_neg_{}'.format(len(self.yi))
        yi_pos = self.model.addVar(lb = 0, name = yi_pos_name, obj = 1)
        if pos_only:
            self.yi.append(yi_pos)
            self.model.update()
            return yi_pos
        yi_neg = self.model.addVar(lb = 0, name = yi_neg_name, obj = 1)
        # make sure new_yi is not already in self.yi
        self.yi.extend([yi_pos, yi_neg])
        self.model.update()
        return yi_pos, yi_neg

    @staticmethod
    def parse_key(key):
        # parsing hackery to get a tuple of ints from its str representation
        return [int(x) for x in key.strip('(').strip(')').strip(' ').split(',')]
    
    def load_constraints_from_file(self, fname, max_constrs=None):
        """
        loads the contraints from the file indicated and adds them to the optimization problem
        """
        MaxMarginModel.load_constraints_from_file(self, fname, max_constrs=max_constrs)
        print 'adding bellman constraints'
        infile = h5py.File(fname, 'r')
        n_added = 0
        n_total = len(infile)
        for key in infile:
            constr = infile[key]            
            if constr['exp_action_name'][()] == 'failure':
                rhs_phi = constr['rhs_phi'][:]
                n_constrs = rhs_phi.shape[0]
                yi_pos_var = self.add_yi(pos_only=True)
                for j in range(n_constrs):
                    self.add_deadend_constraint(rhs_phi[j], yi_pos_var, update=False)
                sys.stdout.write("\rComputed Constraints {}/{}           ".format(n_added, n_total))
                sys.stdout.flush()            
                n_added += 1
                continue
            lhs_phi = constr['exp_phi'][:]
            final_transition = False
            try:
                task_i, step_i = BellmanMaxMarginModel.parse_key(key)
                n_s_i = step_i + 1
                next_constr = infile[str((task_i, n_s_i))]
                rhs_phi = next_constr['exp_phi'][:]
            except KeyError:
                rhs_phi = None
                final_transition=True
            self.add_bellman_constraint(lhs_phi, rhs_phi, update=False, final_transition=final_transition)
            sys.stdout.write("\rComputed Constraints {}/{}           ".format(n_added, n_total))
            sys.stdout.flush()            
            n_added += 1
        self.model.update()
        
    def load_weights_from_file(self, fname):
        infile = h5py.File(fname, 'r')
        self.weights = infile['weights'][:]
        if 'w0' in infile:
            self.w0_val = infile['w0'][()]
        if 'xi' in infile:
            self.xi_val = infile['xi'][()]
        if 'yi' in infile:
            self.yi_val = infile['yi'][()]
        infile.close()
        
    def save_weights_to_file(self, fname):
        # changed to use h5py.File so file i/o is consistent
        outfile = h5py.File(fname, 'w')
        outfile['weights'] = self.weights
        outfile['w0'] = self.w0_val
        outfile['xi'] = self.xi_val
        if self.yi_val:
            outfile['yi'] = self.yi_val
        outfile.close()

    def get_param_fname(self, fname):
        fname_noext = os.path.splitext(fname)[0]
        return fname_noext + '_param.h5'                

    def save_model(self, fname):
        MaxMarginModel.save_model(self, fname)
        param_fname = self.get_param_fname(fname)
        param_f = h5py.File(param_fname, 'w')
        param_f['gamma'] = self.gamma
        param_f['action_reward'] = self.action_reward
    
    def scale_objective(self, C, D):
        self.model.update()
        for xi_var in self.xi:
            xi_var.Obj = C
        for yi_var in self.yi:
            yi_var.Obj = D
        self.model.update()
    
    def optimize_model(self):
        self.model.update()
        self.model.optimize()
        try:
            self.weights = [x.X for x in self.w]
            self.w0_val = self.w0.X
            self.xi_val = [x.X for x in self.xi]
            self.yi_val = [x.X for x in self.yi]
            return self.weights, self.w0_val
        except grb.GurobiError:
            raise RuntimeError, "issue with optimizing model, check gurobi optimizer output"

##TODO: implement unit tests
