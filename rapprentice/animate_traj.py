import trajoptpy, openravepy
import sys

def animate_traj(traj, robot, pause=True, step_viewer=1, restore=True, callback=None, execute_step_cond=None):
    """make sure to set active DOFs beforehand"""
    if restore: _saver = openravepy.RobotStateSaver(robot)
    if step_viewer or pause: viewer = trajoptpy.GetViewer(robot.GetEnv())
    for (i,dofs) in enumerate(traj):
        sys.stdout.write("step %i/%i\r"%(i+1,len(traj)))
        sys.stdout.flush()
        if callback is not None: callback(i)
        if execute_step_cond is not None and not execute_step_cond(i): continue
        robot.SetActiveDOFValues(dofs)
        if pause: viewer.Idle()
        elif step_viewer!=0 and (i%step_viewer==0 or i==len(traj)-1): viewer.Step()
    sys.stdout.write("\n")
