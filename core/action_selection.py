from __future__ import division

from constants import EXACT_LAMBDA, N_ITER_EXACT
import numpy as np
import simulation_object, sim_util
from mmqe.search import beam_search, beam_search_parallel
from rapprentice import registration
from rapprentice.knot_classifier import isKnot as is_knot
from registration import TpsRpmBijRegistrationFactory
from transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from registration_transfer import TwoStepRegistrationAndTrajectoryTransferer
from core.transfer_simulate import BatchTransferSimulate

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
    def plan_agenda(self, scene_state, timestep=-1):
        action2q_value = self.registration_factory.batch_cost(scene_state)
        q_values, agenda = zip(*sorted([(q_value, action) for (action, q_value) in action2q_value.items()]))
        return agenda, q_values

class FeatureActionSelection(ActionSelection):
    def __init__(self, registration_factory, features, actions, demos,
                 width, depth, simulator=None, lfd_env=None, debug=False):
        self.features = features
        self.actions = actions.keys()
        self.features.set_name2ind(self.actions)
        self.demos = demos
        self.width = width
        self.depth = depth
        self.transferer = simulator
        self.lfd_env = lfd_env
        self.debug = debug
        super(FeatureActionSelection, self).__init__(registration_factory)

    def plan_agenda(self, scene_state, timestep=-1):
        def evaluator(state, ts):
            fv = self.features.features(state,timestep=ts)
            val = np.dot(fv, self.features.weights)
            opt = np.argmax(val)
            return np.dot(fv, self.features.weights)

        def simulate_transfer(state, action, next_state_id):
            aug_traj=self.transferer.transfer(self.demos[action], state, plotting=self.debug)
            
            (feas, misgrasp) = self.lfd_env.execute_augmented_trajectory(aug_traj, step_viewer=self.debug)
            
            result_state = self.lfd_env.observe_scene()

            # Get the rope simulation object and determine if it's a knot
            for sim_obj in self.lfd_env.sim.sim_objs:
                if isinstance(sim_obj, simulation_object.RopeSimulationObject):
                    rope_sim_obj = sim_obj
                    break
            rope_knot = is_knot(rope_sim_obj.rope.GetControlPoints())
            fail = not(feas) or misgrasp
            return (result_state, next_state_id, rope_knot, fail)

        return beam_search(scene_state, timestep, self.actions, simulate_transfer,
                           evaluator, self.lfd_env.sim, width=self.width,
                           depth=self.depth)

class ParallelFeatureActionSelection(ActionSelection):
    """
    uses a different thread to do each expansion
    savings should be ~width of search
    """
    def __init__(self, registration_factory, features, actions, demos,
                 width, depth, args, lfd_env=None):
        self.features = features
        self.actions = actions.keys()
        self.features.set_name2ind(self.actions)
        self.demos = demos
        self.width = width
        self.depth = depth
        ##set up cpu registration/transfer
        self.batch_transfer_simulate = BatchTransferSimulate(args, demos)
        self.lfd_env = lfd_env
        super(ParallelFeatureActionSelection, self).__init__(registration_factory)

    def plan_agenda(self, scene_state, timestep=-1):
        def evaluator(state, ts):
            score = np.dot(self.features.features(state, timestep=ts), self.features.weights) + self.features.w0
            # if np.max(score) > -.2:
            #     import ipdb; ipdb.set_trace()
            return score

        return beam_search_parallel(scene_state, timestep, self.actions, self.batch_transfer_simulate,
                           evaluator, self.lfd_env.sim, width=self.width,
                           depth=self.depth, debug=True)

def transfer(transferer, demo, state):
    return transferer.transfer(demo, state, plotting=False)
