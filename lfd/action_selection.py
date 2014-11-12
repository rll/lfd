from __future__ import division

import numpy as np

from lfd.environment import simulation_object
from lfd.mmqe.search import beam_search
from lfd.rapprentice.knot_classifier import isKnot as is_knot


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
    def plan_agenda(self, scene_state, timestep):
        action2q_value = self.registration_factory.batch_cost(scene_state)
        q_values, agenda = zip(*sorted([(q_value, action) for (action, q_value) in action2q_value.items()]))
        # Return false for goal not found
        return (agenda, q_values), False

class FeatureActionSelection(ActionSelection):
    def __init__(self, registration_factory, features, actions, demos,
                 width, depth, simulator=None, lfd_env=None):
        self.features = features
        self.actions = actions.keys()
#        self.features.set_name2ind(self.actions)
        self.demos = demos
        self.width = width
        self.depth = depth
        self.transferer = simulator
        self.lfd_env = lfd_env
        super(FeatureActionSelection, self).__init__(registration_factory)

    def plan_agenda(self, scene_state, timestep):
        def evaluator(state, ts):
            try:
                score = np.dot(self.features.features(state, timestep=ts), self.features.weights) + self.features.w0
            except:
                return -np.inf*np.r_[np.ones(len(self.features.weights))]
            # if np.max(score) > -.2:
            #     import ipdb; ipdb.set_trace()
            return score

        def simulate_transfer(state, action, next_state_id):
            aug_traj=self.transferer.transfer(self.demos[action], state, plotting=False)
            self.lfd_env.execute_augmented_trajectory(aug_traj, step_viewer=0)
            result_state = self.lfd_env.observe_scene()

            # Get the rope simulation object and determine if it's a knot
            for sim_obj in self.lfd_env.sim.sim_objs:
                if isinstance(sim_obj, simulation_object.RopeSimulationObject):
                    rope_sim_obj = sim_obj
                    break
            rope_knot = is_knot(rope_sim_obj.rope.GetControlPoints())
            return (result_state, next_state_id, rope_knot)

        return beam_search(scene_state, timestep, self.features.src_ctx.seg_names, simulate_transfer,
                           evaluator, self.lfd_env.sim, width=self.width,
                           depth=self.depth)
