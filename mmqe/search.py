"""
classes and methods to do a tree search to find the best option
"""

import numpy as np

class SearchNode(object):

    id_map = {}
    action2ind = None
    ind2action = None
    _id = 0
    
    def __init__(self, ID, state, child_vals, child_ids=None, parent = None):
        self.ID               = ID
        self.state            = state
        self.child_vals       = child_vals
        self.value            = None
        SearchNode.id_map[ID] = self
        self.n_expands        = 0
        self.child_expands    = np.zeros(child_vals.shape[0])
        
        if child_ids is None:
            self.child_ids = [SearchNode.get_UID() for _ in child_vals]
        else:
            self.child_ids = child_ids

        self.child_id2ind = dict([(C_ID, i) for i, C_ID in enumerate(self.child_ids)])
        

        if parent is None:
            self.parent = self
        else:
            self.parent = parent        

    def best_k(self, k):
        """
        returns the best k children for this node.
        returns a list of tuples (v, child_a, node_id)
        """    
        best_k = sorted(zip(self.child_vals, SearchNode.ind2action), key=lambda x: -x[0])[:k]
        return [(v, a, self.ID) for (v, a) in best_k]

    def update(self, v, c_id):
        """
        updates this node's value based on the updated value for the child c_id
        """
        c_ind = self.child_id2ind[c_id]
        self.child_vals[c_ind] = v
        self.child_expands[c_ind] += 1


    @staticmethod
    def set_actions(ind2action):
        SearchNode.ind2action = ind2action
        SearchNode.action2ind = dict([(a, i) for i, a in enumerate(ind2action)])

    @staticmethod
    def get_UID():
        uid = SearchNode._id
        SearchNode._id += 1
        return uid

class ExpandingNode(SearchNode):
    """
    placeholder to cover nodes that are currently being expanded
    """
    def __init__(self, ID, parent):
        self.ID               = ID
        self.parent           = parent
        SearchNode.id_map[ID] = self

class MaxNode(SearchNode):
    def __init__(self, ID, state, child_vals, child_ids=None, parent = None):
        SearchNode.__init__(self, ID, state, child_vals, child_ids=child_ids, parent=parent)
        self.value = np.max(self.child_vals)

    def update(self, v, c_id):
        SearchNode.update(self, v, c_id)
        expanded_inds = self.child_expands == np.max(self.child_expands)
        old_value = self.value
        self.value = np.max(self.child_vals[expanded_inds])
        if self.parent != self:
            self.parent.update(self.value, self.ID)

    def select_best(self):
        expanded_inds = self.child_expands == np.max(self.child_expands)
        shifted_vals = self.child_vals + np.max(self.child_vals) * (expanded_inds - 1)
        return [SearchNode.ind2action[np.argmax(shifted_vals)]], [np.max(self.child_vals)]

# env is for resetting the state at each step
def beam_search(start_state, timestep, actions, expander, evaluator, sim, width=1, depth=1):
    id2simstate = {}
    SearchNode.set_actions(actions)
    root_id = SearchNode.get_UID()
    id2simstate[root_id] = sim.get_state()
    root_vals = evaluator(start_state, timestep)
    root = MaxNode(root_id, start_state, root_vals)
    agenda = [root]
    goal_found = False
    for d in range(depth):
        next_states = []
        for s in agenda:
            if len(agenda) >= 3:
                next_states.extend(s.best_k(width/2))#only expand at most half from the same child
            else:
                next_states.extend(s.best_k(width))#unless there are only a couple options
        agenda = sorted(next_states, key=lambda x:-x[0])[:width]
        expand_res = []
        for v, a, P_ID in agenda:
            print 'parent ID:\t{}'.format(P_ID)
            parent_node = SearchNode.id_map[P_ID]
            parent_state = parent_node.state
            child_id = parent_node.child_ids[SearchNode.action2ind[a]]
            sim.set_state(id2simstate[P_ID])
            expand_res.append(expander(parent_state, a, child_id))
            id2simstate[child_id] = sim.get_state()
            child_node = ExpandingNode(child_id, parent_node)
        agenda = []
        for res in expand_res:
            next_s, next_s_id, is_goal = res
            parent = SearchNode.id_map[next_s_id].parent
            del SearchNode.id_map[next_s_id]

            if is_goal:
                goal_found = True
                parent.update(np.inf, next_s_id)
                break
            #elif not res.feasible or res.misgrasp:
            #    parent.update(-np.inf, next_s_id)
            #    continue
            child_vals = evaluator(next_s, timestep+d+1)
            child_node = MaxNode(next_s_id, next_s, child_vals, parent=parent)
            parent.update(child_node.value, next_s_id)
            agenda.append(child_node)
        if goal_found:
            break
    # Reset back to the original state before returning
    sim.set_state(id2simstate[root_id])
    return root.select_best(), goal_found
