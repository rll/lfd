import numpy as np

def draw_grid(env, f, mins, maxes, xres = .1, yres = .1, zres = .04, color = (1,1,0,1)):
    
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

    nfine = 30
    xcoarse = np.arange(xmin, xmax, xres)
    xmax = xcoarse[-1];
    ycoarse = np.arange(ymin, ymax, yres)
    ymax = ycoarse[-1];
    if zres == -1:
        zcoarse = [(zmin+zmax)/2.]
    else:
        zcoarse = np.arange(zmin, zmax, zres)
        zmax = zcoarse[-1];
    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)
    zfine = np.linspace(zmin, zmax, nfine)
    
    lines = []
    if len(zcoarse) > 1:    
        for x in xcoarse:
            for y in ycoarse:
                xyz = np.zeros((nfine, 3))
                xyz[:,0] = x
                xyz[:,1] = y
                xyz[:,2] = zfine
                lines.append(f(xyz))

    for y in ycoarse:
        for z in zcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = xfine
            xyz[:,1] = y
            xyz[:,2] = z
            lines.append(f(xyz))
        
    for z in zcoarse:
        for x in xcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = x
            xyz[:,1] = yfine
            xyz[:,2] = z
            lines.append(f(xyz))

    handles = []

    for line in lines:
        handles.append(env.drawlinestrip(line,1,color))
                                
    return handles
