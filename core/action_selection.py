from __future__ import division

from constants import EXACT_LAMBDA, N_ITER_EXACT
import numpy as np
from rapprentice import registration

import IPython as ipy

class ActionSelection(object):
    def __init__(self, registration_factory):
        """Inits ActionSelection
        
        Args:
            registration_factory: RegistrationFactory
        """
        self.registration_factory = registration_factory
        
    def plan_agenda(self, scene_state):
        """Plans an agenda of demonstrations for the given scene_state
        
        Args:
            scene_state: SceneState of the scene for which the agenda is returned
        
        Returns:
            An agenda, which is a list of demonstration names, and a list of the values of the respective demonstrations
        """
        raise NotImplementedError

class GreedyActionSelection(ActionSelection):
    def plan_agenda(self, scene_state):
        action2q_value = self.registration_factory.batch_cost(scene_state)
        q_values, agenda = zip(*sorted([(q_value, action) for (action, q_value) in action2q_value.items()]))
        return agenda, q_values

class MmqeActionSelection(ActionSelection):
    # TODO Dylan
    def __init__(self, registration_factory):
        super(MmqeActionSelection, self).__init__(registration_factory)
        raise NotImplementedError
    
    def plan_agenda(self, scene_state):
        raise NotImplementedError

class SoftmaxActionSelection(ActionSelection):
    def __init__(self, registration_factory, alpha=10):
        super(SoftmaxActionSelection, self).__init__(registration_factory)
        self.alpha = alpha

    def plan_agenda(self, scene_state):
        """
        a ~ softmax(q_values; alpha)
        """
        action2q_value   = self.registration_factory.batch_cost(scene_state)
        q_values, agenda = zip(*sorted([(q_value, action) for (action, q_value) in action2q_value.items()]))

        sm_values = np.exp(self.alpha * np.asarray(q_values))
        sm_values = np.cumsum(sm_values / np.sum(sm_values))# CDF
        i_chosen  = np.searchsorted(sm_values, np.random.rand())
        return [agenda[i_chosen]], [q_values[i_chosen]]

class ParentActionSelection(ActionSelection):

    def __init__(self, registration_factory, base_selector):
        super(ParentActionSelection, self).__init__(registration_factory)
        self.base_selector = base_selector

    def plan_agenda(self, scene_state):
        agenda, q_values = self.base_selector.plan_agenda(scene_state)
        selected_demo = self.registration_factory.demos[agenda[0]]
        try:
            while True:
                selected_demo = selected_demo.parent
        except AttributeError:
            pass
        return [selected_demo.name], [q_values[0]]
