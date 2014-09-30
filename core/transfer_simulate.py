from __future__ import division

import sim_util
import h5py
import numpy as np
from simulation import DynamicRopeSimulationRobotWorld
from environment import LfdEnvironment
from registration import TpsRpmBijRegistrationFactory
from transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from registration_transfer import TwoStepRegistrationAndTrajectoryTransferer
from rapprentice.knot_classifier import isKnot as is_knot
from core import simulation_object, sim_util
import sys, os

from IPython import parallel
from IPython.parallel.util import interactive

import IPython as ipy

class BatchTransferSimulate(object):
    def __init__(self, args, demos, max_queue_size = 100, profile='ssh'):
        self.max_queue_size = max_queue_size

        # create clients and views
        self.rc = parallel.Client(profile=profile)
        self.dv = self.rc[:]
        self.v = self.rc.load_balanced_view()
 
        # add module paths to the engine paths
        # modules = ['lfd']
        # module_paths = []
        # for module in modules:
        #     paths = [path for path in sys.path if module == os.path.split(path)[1]]
        #     assert len(paths) > 0
        #     module_paths.append(paths[0]) # add the first module path only
        # module_paths=['~/src/lfd']
        # @interactive
        # def engine_add_module_paths(module_paths):
        #     import sys
        #     sys.path.extend(module_paths)
        # self.dv.map_sync(engine_add_module_paths, [module_paths]*len(self.dv))

        @interactive
        def engine_initialize(id, args, demos):
            from core.simulation import DynamicRopeSimulationRobotWorld
            from core.environment import LfdEnvironment
            from core.registration import TpsRpmBijRegistrationFactory
            from core.transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
            from core.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer
            global lfd_env, reg_and_traj_transferer
            sim = DynamicRopeSimulationRobotWorld()
            sim_transfer = DynamicRopeSimulationRobotWorld()
            world = sim
            lfd_env = LfdEnvironment(sim, world, downsample_size=args.eval.downsample_size)
            #reg_factory = None
            reg_factory = TpsRpmBijRegistrationFactory(demos)
            traj_transferer = PoseTrajectoryTransferer(sim_transfer, args.eval.beta_pos, args.eval.beta_rot, 
                                                       args.eval.gamma, args.eval.use_collision_cost)
            traj_transferer = FingerTrajectoryTransferer(sim_transfer, args.eval.beta_pos, args.eval.gamma, 
                                                         args.eval.use_collision_cost, 
                                                         init_trajectory_transferer=traj_transferer)
            reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
        self.dv.map_sync(engine_initialize, self.rc.ids, [args]*len(self.dv.targets), [demos]*len(self.dv.targets))
        sim = DynamicRopeSimulationRobotWorld()
        world = sim
        lfd_env = LfdEnvironment(sim, world, downsample_size=args.eval.downsample_size)
        reg_factory = TpsRpmBijRegistrationFactory(demos)
        traj_transferer = PoseTrajectoryTransferer(sim, args.eval.beta_pos, args.eval.beta_rot, 
                                                   args.eval.gamma, args.eval.use_collision_cost)
        traj_transferer = FingerTrajectoryTransferer(sim, args.eval.beta_pos, args.eval.gamma, 
                                                     args.eval.use_collision_cost, 
                                                     init_trajectory_transferer=traj_transferer)
        reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
        self.lfd_env = lfd_env
        self.reg_and_traj_transferer = reg_and_traj_transferer
        self.pending = set()
        
    def queue_transfer_simulate(self, simstate, state, action, next_state_id): # TODO optional arguments
        self.wait_while_queue_is_full()
        @interactive
        def engine_transfer_simulate(simstate, state, action, metadata):            
            from rapprentice.knot_classifier import isKnot as is_knot
            from core import simulation_object, sim_util
            global lfd_env, reg_and_traj_transferer
            lfd_env.sim.set_state(simstate)
            demo = reg_and_traj_transferer.registration_factory.demos[action]
            aug_traj = reg_and_traj_transferer.transfer(demo, state, simstate, plotting=False)
            (feas, misgrasp) = lfd_env.execute_augmented_trajectory(aug_traj, step_viewer=0)
            lfd_env.sim.settle()
            result_state = lfd_env.observe_scene()
            for sim_obj in lfd_env.sim.sim_objs:
                if isinstance(sim_obj, simulation_object.RopeSimulationObject):
                    rope_sim_obj = sim_obj
                    break
            rope_knot = is_knot(rope_sim_obj.rope.GetControlPoints())
            fail = not(feas) or misgrasp or result_state.cloud.shape[0] < 10
            return {'result_state': result_state,
		    'action': action, 
                    'metadata': metadata, 
                    'is_knot':rope_knot, 
                    'is_failure':fail, 
                    'next_simstate': lfd_env.sim.get_state(), 
                    'aug_traj': aug_traj}

        amr = self.v.map(engine_transfer_simulate, *[[e] for e in [simstate, state, action, next_state_id]])
        self.pending.update(amr.msg_ids)

    def queue_transfer(self, simstate, state, action): # TODO optional arguments
        self.wait_while_queue_is_full()
        @interactive
        def engine_transfer_simulate(simstate, state, action):            
            from rapprentice.knot_classifier import isKnot as is_knot
            from core import simulation_object, sim_util
            global lfd_env, reg_and_traj_transferer
            lfd_env.sim.set_state(simstate)
            demo = reg_and_traj_transferer.registration_factory.demos[action]
            aug_traj = reg_and_traj_transferer.transfer(demo, state, simstate, plotting=False)
            return (aug_traj, simstate, action)

        amr = self.v.map(engine_transfer_simulate, *[[e] for e in [simstate, state, action]])
        self.pending.update(amr.msg_ids)


    def wait_while_queue_size_above_size(self, queue_size):
        pending = self.pending.copy()
        while len(pending) > queue_size:
            try:
                self.rc.wait(pending, 1e-3)
            except parallel.TimeoutError:
                # ignore timeouterrors, since they only mean that at least one isn't done
                pass
            # finished is the set of msg_ids that are complete
            finished = pending.difference(self.rc.outstanding)
            # update pending to exclude those that just finished
            pending = pending.difference(finished)

    def wait_while_queue_is_full(self):
        self.wait_while_queue_size_above_size(self.max_queue_size)

    def wait_while_queue_is_nonempty(self):
        self.wait_while_queue_size_above_size(0)

    def get_results(self):
        results = []
        try:
            self.rc.wait(self.pending, 1e-3)
        except parallel.TimeoutError:
            # ignore timeouterrors, since they only mean that at least one isn't done
            pass
        # finished is the set of msg_ids that are complete
        finished = self.pending.difference(self.rc.outstanding)
        # update pending to exclude those that just finished
        self.pending = self.pending.difference(finished)
        for msg_id in finished:
            # we know these are done, so don't worry about blocking
            ar = self.rc.get_result(msg_id)
            results.extend(ar.result)
        return results
