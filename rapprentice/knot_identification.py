from collections import defaultdict
import multiprocessing
import numpy as np
from rapprentice import math_utils, LOG

def intersect_segs(ps_n2, q_22):
    """Takes a list of 2d nodes (ps_n2) of a piecewise linear curve and two points representing a single segment (q_22)
        and returns indices into ps_n2 of intersections with the segment."""
    assert ps_n2.shape[1] == 2 and q_22.shape == (2, 2)

    def cross(a_n2, b_n2):
        return a_n2[:,0]*b_n2[:,1] - a_n2[:,1]*b_n2[:,0]

    rs = ps_n2[1:,:] - ps_n2[:-1,:]
    s = q_22[1,:] - q_22[0,:]
    denom = cross(rs, s[None,:])
    qmp = q_22[0,:][None,:] - ps_n2[:-1,:]
    ts = cross(qmp, s[None,:]) / denom # zero denom will make the corresponding element of 'intersections' false
    us = cross(qmp, rs) / denom # same here
    intersections = np.flatnonzero((ts > 0) & (ts < 1) & (us > 0) & (us < 1))
    return intersections, ts, us

def rope_has_intersections(ctl_pts):
    for i in range(len(ctl_pts) - 1):
        curr_seg = ctl_pts[i:i+2,:]
        intersections, ts, us = intersect_segs(ctl_pts[:,:2], curr_seg[:,:2])
        if len(intersections) != 0:
            return True
    return False

def compute_dt_code(ctl_pts, plotting=False):
    """Takes rope control points (Nx3 array), closes the loop, and computes the Dowker-Thistlethwaite code for the knot.
       The z-value for the points are used for determining over/undercrossings.
       Follows procedure outlined here: http://katlas.math.toronto.edu/wiki/DT_(Dowker-Thistlethwaite)_Codes
       """

    # First, close the loop by introducing extra points under the table and toward the robot (by subtracting z and x values)
    # first_pt, last_pt =  ctl_pts[0], ctl_pts[-1]
    # flipped = False
    # if first_pt[1] > last_pt[1]:
    #     first_pt, last_pt = last_pt, first_pt
    #     flipped = True
    # min_z = ctl_pts[:,2].min()
    # extra_first_pt, extra_last_pt = first_pt + [-.1, -.1, min_z-1], last_pt + [-.1, .1, min_z-1]
    # if flipped:
    #     extra_pts = [extra_first_pt, extra_first_pt + [-1, 0, 0], extra_last_pt + [-1, 0, 0], extra_last_pt, last_pt]
    # else:
    #     extra_pts = [extra_last_pt, extra_last_pt + [-1, 0, 0], extra_first_pt + [-1, 0, 0], extra_first_pt, first_pt]
    # ctl_pts = np.append(ctl_pts, extra_pts, axis=0)

    if plotting:
        import trajoptpy, openravepy
        env = openravepy.Environment()
        viewer = trajoptpy.GetViewer(env)
        handles = []
        handles.append(env.plot3(ctl_pts, 5, [0, 0, 1]))
        viewer.Idle()

    # Upsampling loop: upsample until every segment has at most one crossing
    need_upsample_ind = None
    upsample_iters = 0
    max_upsample_iters = 10
    while True:
        counter = 1
        crossings = defaultdict(list)
        # Walk along rope: for each segment, compute intersections with all other segments
        for i in range(len(ctl_pts) - 1):
            curr_seg = ctl_pts[i:i+2,:]
            intersections, ts, us = intersect_segs(ctl_pts[:,:2], curr_seg[:,:2])

            if len(intersections) == 0:
                continue
            if len(intersections) != 1:
                LOG.debug('warning: more than one intersection for segment %d, now upsampling', i)
                need_upsample_ind = i
                break

            # for each intersection, determine and record over/undercrossing
            i_int = intersections[0]
            if plotting:
                handles.append(env.drawlinestrip(ctl_pts[i_int:i_int+2], 5, [1, 0, 0]))
            int_point_rope = ctl_pts[i_int] + ts[i_int]*(ctl_pts[i_int+1] - ctl_pts[i_int])
            int_point_curr_seg = curr_seg[0] + us[i_int]*(curr_seg[1] - curr_seg[0])
            #assert np.allclose(int_point_rope[:2], int_point_curr_seg[:2])
            above = int_point_curr_seg[2] > int_point_rope[2]
            crossings[tuple(sorted((i, i_int)))].append(-counter if counter % 2 == 0 and above else counter)
            counter += 1
        if plotting: viewer.Idle()
        # upsample if necessary
        if need_upsample_ind is not None and upsample_iters < max_upsample_iters:
            spacing = np.linspace(0, 1, len(ctl_pts))
            new_spacing = np.insert(spacing, need_upsample_ind+1, (spacing[need_upsample_ind]+spacing[need_upsample_ind+1])/2.)
            ctl_pts = math_utils.interp2d(new_spacing, spacing, ctl_pts)
            upsample_iters += 1
            need_upsample = None
            continue
        break

    # Extract relevant part of crossing data to produce DT code
    out = []
    for pair in crossings.itervalues():
        assert len(pair) == 2
        odd = [p for p in pair if p % 2 == 1][0]
        even = [p for p in pair if p % 2 == 0][0]
        out.append((odd, even))
    out.sort()
    dt_code = [-o[1] for o in out]
    return dt_code

def _dt_code_to_knot(dt_code):
    import snappy
    try:
        m = snappy.Manifold("DT:[%s]" % ",".join(map(str, dt_code)))
        knot = snappy.HTLinkExteriors.identify(m)
        return knot.name()
    except:
        import traceback
        traceback.print_exc()
        return None


def dt_code_to_knot(dt_code):
    def dt_code_to_knot_wrapper(q, x):
        result = _dt_code_to_knot(x)
        q.put(result)
        q.close()

    q = multiprocessing.Queue(1)
    proc = multiprocessing.Process(target=dt_code_to_knot_wrapper, args=(q, dt_code))
    proc.start()
    TIMEOUT = 1
    try:
        result = q.get(True, TIMEOUT)
    except:
        LOG.warn("Timeout for knot identification exceeded, assuming no knot")
        result = None
    finally:
        proc.terminate()
    return result

def identify_knot(ctl_pts):
    """Given control points from a rope, gives a knot name if identified by snappy, or None otherwise"""

    try:
        dt_code = compute_dt_code(ctl_pts)
        print 'dt code', dt_code
        return dt_code_to_knot(dt_code)
    except:
        import traceback
        traceback.print_exc()
        return None

def main():
    #dt_code = [8, 6, -4, -10, 2]
    #dt_code = [4, 6, 2, -10, 8]
    dt_code = [4, 6, 2, -8]
    # m = snappy.Manifold("DT:[%s]" % ",".join(map(str, dt_code)))
    # knot = snappy.HTLinkExteriors.identify(m)
    # print knot.name()
    #print dt_code_to_knot(dt_code)
    #return

    import cPickle
    with open("results/single_example_no_failures_100_03cm_s0.pkl", "r") as f: experiments = cPickle.load(f)
    log = experiments[2][1]
    rope_nodes = []
    for entry in log:
        if 'sim_rope_nodes_after_full_traj' in entry.name:
            rope_nodes.append(entry.data)

    for i, n in enumerate(rope_nodes):
        knot = identify_knot(n)
        print "[%d/%d] %s" % (i+1, len(rope_nodes), knot)

if __name__ == '__main__':
    main()
