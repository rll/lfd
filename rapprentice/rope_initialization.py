import networkx as nx, numpy as np, scipy.spatial.distance as ssd, scipy.interpolate as si
from collections import deque
import itertools
from numpy.random import rand
from rapprentice import knot_identification



########### TOP LEVEL FUNCTION ###############

MIN_SEG_LEN = 3

def find_path_through_point_cloud(xyzs, plotting=False, perturb_peak_dist=None, num_perturb_points=7):
    xyzs = np.asarray(xyzs).reshape(-1,3)
    S = skeletonize_point_cloud(xyzs)
    segs = get_segments(S)

    if plotting: 
        from mayavi import mlab
        mlab.figure(1); mlab.clf()
        plot_graph_3d(S)

    S,segs = prune_skeleton(S, segs)

    if plotting: 
        from mayavi import mlab
        mlab.figure(3); mlab.clf()
        plot_graph_3d(S)


    segs3d = [np.array([S.node[i]["xyz"] for i in seg]) for seg in segs]

    if plotting: plot_paths_2d(segs3d)

    C = make_cost_matrix(segs3d)
    PG = make_path_graph(C, [len(path) for path in segs3d])

    (score, nodes) = longest_path_through_segment_graph(PG)
#    print nodes

    total_path = []
    for node in nodes[::2]:
        if node%2 == 0: total_path.extend(segs[node//2])
        else: total_path.extend(segs[node//2][::-1])

    total_path_3d = remove_duplicate_rows(np.array([S.node[i]["xyz"] for i in total_path]))

    # perturb the path, if requested
    if perturb_peak_dist is not None:
        orig_path_length = np.sqrt(((total_path_3d[1:,:2] - total_path_3d[:-1,:2])**2).sum(axis=1)).sum()
        perturb_centers = np.linspace(0, len(total_path_3d)-1, num_perturb_points).astype(int)
        perturb_xy = np.zeros((len(total_path_3d), 2))
        bandwidth = len(total_path_3d) / (num_perturb_points-1)

        # add a linearly decreasing peak around each perturbation center
        # (keep doing so randomly until our final rope has no loops)
        for _ in range(20):
            for i_center in perturb_centers:
                angle = np.random.rand() * 2 * np.pi
                range_min = max(0, i_center - bandwidth)
                range_max = min(len(total_path_3d), i_center + bandwidth + 1)

                radii = np.linspace(0, perturb_peak_dist, i_center+1-range_min)
                perturb_xy[range_min:i_center+1,:] += np.c_[radii*np.cos(angle), radii*np.sin(angle)]

                radii = np.linspace(perturb_peak_dist, 0, range_max-i_center)
                perturb_xy[i_center+1:range_max,:] += np.c_[radii*np.cos(angle), radii*np.sin(angle)][1:,:]

            unscaled_path_2d = total_path_3d[:,:2] + perturb_xy
            if not knot_identification.rope_has_intersections(unscaled_path_2d):
                break

        # re-scale to match path length of original rope
        center = np.mean(unscaled_path_2d, axis=0)
        lo, curr, hi = 0, 1, 10
        for _ in range(10):
            tmp_path_2d = curr*(unscaled_path_2d - center) + center
            path_length = np.sqrt(((tmp_path_2d[1:] - tmp_path_2d[:-1])**2).sum(axis=1)).sum()
            if abs(path_length - orig_path_length) < .01:
                break
            if path_length > orig_path_length:
                hi = curr
            elif path_length < orig_path_length:
                lo = curr
            curr = (lo + hi) / 2.
        total_path_3d[:,:2] = tmp_path_2d

    total_path_3d = unif_resample(total_path_3d, seg_len=.02, tol=.003) # tolerance of 1mm
    total_path_3d = unif_resample(total_path_3d, seg_len=.02, tol=.003) # tolerance of 1mm
#    total_path_3d = unif_resample(total_path_3d, seg_len=.02, tol=.003) # tolerance of 1mm

    if plotting:
        mlab.figure(2); mlab.clf()
        for seg in segs3d:
            x,y,z = np.array(seg).T
            mlab.plot3d(x,y,z,color=(0,1,0),tube_radius=.001)
        x,y,z = np.array(total_path_3d).T
        mlab.plot3d(x,y,z,color=(1,0,0),tube_radius=.01,opacity=.2)      
        
        x,y,z = np.array(xyzs).reshape(-1,3).T
        mlab.points3d(x,y,z,scale_factor=.01,opacity=.1,color=(0,0,1))


    return total_path_3d
    
def prune_skeleton(S, segs):
    bad_segs = []
    bad_nodes = []
    for seg in segs:
        if len(seg) < MIN_SEG_LEN:# and 1 in [S.degree(node) for node in seg]:
            bad_segs.append(seg)
            for node in seg:
                if S.degree(node) <= 2: bad_nodes.append(node)
    for seg in bad_segs: segs.remove(seg)
    for node in bad_nodes: S.remove_node(node)
    return S, segs
    
    
def get_skeleton_points(xyzs):
    S = skeletonize_point_cloud(xyzs)
    points = np.array([S.node[i]["xyz"] for i in S.nodes()])
    return points
    

def plot_paths_2d(paths3d):
    import matplotlib.pyplot as plt; plt.ion()
    ax = plt.gca()
    for (i,path) in enumerate(paths3d):
        plt.plot(path[:,0], path[:,1])
        (dx, dy) = rand(2)*.01
        (x,y) = path[0,0], path[0,1]
        ax.annotate(str(2*i), xy=(x,y), xytext=(x+dx, y+dy))
        (dx, dy) = rand(2)*.01
        (x,y) = path[-1,0], path[-1,1]
        ax.annotate(str(2*i+1), xy=(x,y), xytext=(x+dx, y+dy))
    plt.draw()
    
def unif_resample(x,n=None,tol=0,deg=None, seg_len = .02):
 
    if deg is None: deg = min(3, len(x) - 1)

    x = np.atleast_2d(x)
    x = remove_duplicate_rows(x)
    
    (tck,_) = si.splprep(x.T,k=deg,s = tol**2*len(x),u=np.linspace(0,1,len(x)))
    xup = np.array(si.splev(np.linspace(0,1, 10*len(x),.1),tck)).T
    dl = norms(xup[1:] - xup[:-1],1)
    l = np.cumsum(np.r_[0,dl])
    (tck,_) = si.splprep(xup.T,k=deg,s = tol**2*len(xup),u=l)


    if n is not None: newu = np.linspace(0,l[-1],n)
    else: newu = np.linspace(0, l[-1], l[-1]//seg_len)
    return np.array(si.splev(newu,tck)).T    

############## SKELETONIZATION ##############


def points_to_graph(xyzs, max_dist):
    pdists = ssd.squareform(ssd.pdist(xyzs))
    G = nx.Graph()
    for (i_from, row) in enumerate(pdists):
        G.add_node(i_from, xyz = xyzs[i_from])
        to_inds = np.flatnonzero(row[:i_from] < max_dist)
        for i_to in to_inds:
            G.add_edge(i_from, i_to, length = pdists[i_from, i_to])
    return G

def skeletonize_graph(G, resolution):
    #G = largest_connected_component(G)
    partitions = []
    for SG in nx.connected_component_subgraphs(G):
        calc_distances(SG)
        partitions.extend(calc_reeb_partitions(SG,resolution))
    node2part = {}
    skel = nx.Graph()    
    for (i_part, part) in enumerate(partitions):
        xyzsum = np.zeros(3)
        for node in part:
            node2part[node] = i_part
            xyzsum += G.node[node]["xyz"]
        skel.add_node(i_part,xyz = xyzsum / len(part))
            
    for (node0, node1) in G.edges():
        if node2part[node0] != node2part[node1]:
            skel.add_edge(node2part[node0], node2part[node1])
        
    return skel

def largest_connected_component(G):
    sgs = nx.connected_component_subgraphs(G)
    return sgs[0]

def skeletonize_point_cloud(xyzs, point_conn_dist = .025, cluster_size = .04):
    G = points_to_graph(xyzs, point_conn_dist)
    S = skeletonize_graph(G, cluster_size)
    return S
        
def calc_distances(G):
    node0 = G.nodes_iter().next()
    G.node[node0]["dist"] = 0
    frontier = deque([node0])
    
    while len(frontier) > 0:
        node = frontier.popleft()
        node_dist = G.node[node]["dist"]
        for nei in G.neighbors_iter(node):
            nei_dist = G.node[nei].get("dist")
            if G.node[nei].get("dist") is None or G.edge[node][nei]["length"] + node_dist < nei_dist:
                frontier.append(nei)
                G.node[nei]["dist"] = G.edge[node][nei]["length"] + node_dist
        
def calc_reeb_partitions(G, resolution):
    nodes = G.nodes()
    distances = np.array([G.node[node]["dist"] for node in nodes])
    bin_edges = np.arange(distances.min()-1e-5, distances.max()+1e-5, resolution)
    bin_inds = np.searchsorted(bin_edges, distances) - 1
    dist_partitions =  [[] for _ in xrange(bin_inds.max() + 1)]
    for (i_node,i_part) in enumerate(bin_inds):
        dist_partitions[i_part].append(nodes[i_node])
        
    reeb_partitions = []
    for part in dist_partitions:
        sg = G.subgraph(part)
        reeb_partitions.extend(nx.connected_components(sg))
    return reeb_partitions
    
def plot_graph_3d(G):
    from mayavi import mlab
    for (node0, node1) in G.edges():
        (x0,y0,z0) = G.node[node0]["xyz"]
        (x1,y1,z1) = G.node[node1]["xyz"]
        mlab.plot3d([x0,x1],[y0,y1],[z0,z1], color=(1,0,0),tube_radius=.0025)        
    
    
############### GENERATE PATHS FROM GRAPH ###########

def get_segments(G):
    segments = []
    for SG in nx.connected_component_subgraphs(G):
        segments.extend(get_segments_connected(SG))
    return segments
def get_segments_connected(G):
    segments = []
    for junc in get_junctions(G):
        for nei in G.neighbors(junc):
            seg = get_segment_from_junction(G,junc, nei)
            if seg[0] < seg[-1]:
                segments.append(seg)
            if seg[0] == seg[-1] and seg[::-1] not in segments:
                segments.append(seg)
                
    return segments
def get_junctions(G):
    return [node for (node,deg) in G.degree().items() if deg != 2]
def get_segment_from_junction(G, junc, nei):
    seg = [junc, nei]
    while G.degree(seg[-1]) == 2:
        for node in G.neighbors(seg[-1]):
            if node != seg[-2]:
                seg.append(node)
                break
    return seg
    
 
    
    
############### LONGEST PATH STUFF ##############

MAX_COST = 50
SKIP_COST = 1000 # per meter
MAX_NEIGHBORS=5

WEIGHTS = np.r_[
    3, # ang_a_disp: angle between end of source and displacement targ-source
    3, # ang_b_disp: angle between end of target and tisplacement source-targ
    30, # ang_a_b: angle between end of source and end of targ
    30,# fwddist_b_a: positive part of displacement `dot` source direction
    50,# backdist_b_a: negative part of ...
    50,# perpdist_b_a: displacement perpendicular to source direction
    30,# fwddist_a_b
    50,# fwddist_a_b
    50] # perpdist_a_b


def calc_path_features(pos_a, dir_a, pos_b, dir_b):
    "dir_a points away from a"
    ang_a_disp = ang_between(dir_a, pos_a - pos_b)
    ang_b_disp = ang_between(dir_b, pos_b - pos_a)
    ang_a_b = ang_between(-dir_a, dir_b)
    fwddist_b_a = np.clip(np.dot(pos_b - pos_a, -dir_a),0,np.inf)
    backdist_b_a = -np.clip(np.dot(pos_b - pos_a, -dir_a),-np.inf,0)
    perpdist_b_a = np.sqrt(sqnorm(pos_b - pos_a) - np.dot(pos_b - pos_a, -dir_a)**2)
    fwddist_a_b = np.clip(np.dot(pos_a - pos_b, -dir_b),0,np.inf)
    backdist_a_b = -np.clip(np.dot(pos_a - pos_b, -dir_b),-np.inf,0)    
    perpdist_a_b = np.sqrt(sqnorm(pos_a - pos_b) - np.dot(pos_a - pos_b, -dir_b)**2)
    return np.array([
        ang_a_disp,
        ang_b_disp,
        ang_a_b,
        fwddist_b_a,
        backdist_b_a,
        perpdist_b_a,
        fwddist_a_b,
        backdist_a_b,
        perpdist_a_b])

def sqnorm(x):
    return (x**2).sum()
norm = np.linalg.norm    
def ang_between(x,y):
    return np.arccos(np.dot(x,y)/norm(x)/norm(y))
def normalized(x):
    return x / norm(x)
def start_pos_dir(path):
    return path[0], normalized(path[min(len(path)-1,5)] - path[0])
def end_pos_dir(path):
    return start_pos_dir(path[::-1])

def estimate_end_directions(points, tol):
    points = np.asarray(points)
    n = len(points)
    deg = min(3, n - 1)
    u = np.arange(n)
    (tck, _) = si.splprep(points.T, s=tol**2*n, u=u, k=deg)    
    start_dir = np.array(si.splev(u[0],tck,der=1)).T
    end_dir = - np.array(si.splev(u[-1],tck,der=1)).T
    return normalized(start_dir), normalized(end_dir)

def make_cost_matrix(paths):
    pos_and_dirs = []
    for path in paths:
        start_dir, end_dir = estimate_end_directions(path,.01)
        start_pos = path[0]
        end_pos = path[-1]
        pos_and_dirs.append((start_pos, start_dir))
        pos_and_dirs.append((end_pos, end_dir))

    N = len(paths)
    cost_matrix = np.zeros((2*N, 2*N))
    for (i,j) in itertools.combinations(range(2*N),2):
        if j==i+1 and i%2==0:
            cost_matrix[i,j] = cost_matrix[j,i] = np.inf
        else:            
            pdi = pos_and_dirs[i]
            pdj = pos_and_dirs[j]
            features = calc_path_features(pdi[0],pdi[1], pdj[0],pdj[1])
            features[np.isnan(features)] = 0
            cost_matrix[i,j] = cost_matrix[j,i] =  np.dot(WEIGHTS, features)
        #cost_matrix[i,j] = cost_matrix[j,i] =  np.linalg.norm(pdi[0] - pdj[0])*1000
#    print cost_matrix
    return cost_matrix    

def make_path_graph(cost_matrix, lengths):
    M,N = cost_matrix.shape
    assert M==N and M%2==0    

    path_graph = nx.Graph()
    for i in xrange(N):
        sortinds = cost_matrix[i].argsort()
        for j in sortinds[:MAX_NEIGHBORS]:
            if i!=j and cost_matrix[i,j] < MAX_COST:
                path_graph.add_edge(i,j,weight = -cost_matrix[i,j])
    for i in xrange(0,N,2):
        if not path_graph.has_node(i): path_graph.add_node(i)
        if not path_graph.has_node(i+1): path_graph.add_node(i+1)
        path_graph.add_edge(i,i+1,weight=lengths[i//2]*SKIP_COST)
    return path_graph


def longest_path_between(G,start,used,target):  
    
    opp = start - 1 if start%2==1 else start+1
    thislength, thispath = (G.edge[start][opp]["weight"],[start,opp])
    
    if opp == target:
        return (thislength, thispath)
    
    else:
        lengths_paths = []
        newused = used.union([start,opp])
        for nei in G.neighbors_iter(opp):
            if nei not in newused:
                neilength,neipath = longest_path_between(G,nei,newused,target)
                if neilength is not None:
                    lengths_paths.append(
                        (thislength+G.edge[opp][nei]["weight"]+neilength,thispath+neipath))

    if len(lengths_paths) > 0:
        return max(lengths_paths)    
    else: 
        return None,None
    
def longest_path_from(G,start,used):  
    opp = start - 1 if start%2==1 else start+1
    thislength, thispath = (G.edge[start][opp]["weight"],[start,opp])
    lengths_paths = [(thislength, thispath)]
    newused = used.union([start,opp])
    for nei in G.neighbors_iter(opp):
        if nei not in newused:
            neilength,neipath = longest_path_from(G,nei,newused)
            lengths_paths.append(
                (thislength+G.edge[opp][nei]["weight"]+neilength,thispath+neipath))
    return max(lengths_paths)    

def longest_path_through_segment_graph(G):
    best_score = -np.inf
    best_nodes = []

    for i_start in G.nodes():
        (score,nodes) = longest_path_from(G, i_start, set([]))
        if score > best_score:
            best_score = score
            best_nodes = nodes
    return (best_score, best_nodes)


def remove_duplicate_rows(mat):
    diffs = mat[1:] - mat[:-1]
    return mat[np.r_[True,(abs(diffs) >= 1e-5).any(axis=1)]]
def norms(x,ax):
    return np.sqrt((x**2).sum(axis=ax))
