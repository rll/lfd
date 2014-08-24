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
