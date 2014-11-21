import math
import numpy
import random
import operator
import types
import numpy as np

class Transform(object):
    """
    Rotation and translation represented as 4 x 4 matrix
    """
    def __init__(self, matrix):
        self.matrix = numpy.array(matrix)
        self.matrix_inv = None
        self.zRot = False

    def inverse(self):
        """
        Returns transformation matrix that is the inverse of this one
        """
        if self.matrix_inv == None:
            self.matrix_inv =  numpy.linalg.inv(self.matrix)
        return Transform(self.matrix_inv)

    def __neg__(self):
        return self.inverse()

    def compose(self, trans):
        """
        Returns composition of self and trans
        """
        tr = Transform(numpy.dot(self.matrix, trans.matrix))
        if self.zRot and trans.zRot:
            return tr.pose()
        else:
            return tr

    def __mul__(self, other):
        return self.compose(other)

    def pose(self, zthr = 0.01, fail = True):
        """
        Convert to Pose
        """
        if abs(1 - self.matrix[2][2]) < zthr:
            theta = math.atan2(self.matrix[1][0], self.matrix[0][0])
            return Pose(self.matrix[0][3], self.matrix[1][3], self.matrix[2][3], theta)
        elif fail:
            print self.matrix
            raise Exception, "Not a valid 2.5D Pose"
        else:
            return None

    def point(self):
        return self.pose().point()

    def applyToPoint(self, point):
        """
        Transform a point into a new point.
        """
        p = numpy.dot(self.matrix, point.matrix())
        return Point(p[0], p[1], p[2], p[3])

    def __call__(self, point):
        return self.applyToPoint(point)

    def __repr__(self):
        return str(self.matrix)
    def shortStr(self, trim = False):
        return self.__repr__()
    __str__ = __repr__

class Pose(Transform):                  # 2.5D transform
    """
    Represent the x, y, z, theta pose of an object in 2.5D space
    """
    def __init__(self, x, y, z, theta):
        self.x = x
        """x coordinate"""
        self.y = y
        """y coordinate"""
        self.z = z
        """z coordinate"""
        self.theta = fixAngle02Pi(theta)
        """rotation in radians"""
        self.initTrans()
        self.zRot = True

    def initTrans(self):
        cosTh = math.cos(self.theta)
        sinTh = math.sin(self.theta)
        self.reprString = None
        Transform.__init__(self, [[cosTh, -sinTh, 0.0, self.x],
                                  [sinTh, cosTh, 0.0, self.y],
                                  [0.0, 0.0, 1.0, self.z],
                                  [0, 0, 0, 1]])

    def setX(self, x):
        self.x = x
        self.initTrans()

    def setY(self, y):
        self.y = y
        self.initTrans()

    def setZ(self, z):
        self.z = z
        self.initTrans()

    def setTheta(self, theta):
        self.theta = theta
        self.initTrans()

    def average(self, other, alpha):
        """
        Weighted average of this pose and other
        """
        return Pose(alpha * self.x + (1 - alpha) * other.x,
                    alpha * self.y + (1 - alpha) * other.y,
                    alpha * self.z + (1 - alpha) * other.z,
                    angleAverage(self.theta, other.theta, alpha))
        
    def point(self):
        """
        Return just the x, y, z parts represented as a C{Point}
        """
        return Point(self.x, self.y, self.z)

    def pose(self, fail = False):
        return self

    def near(self, pose, distEps, angleEps):
        """
        Return True if pose is within distEps and angleEps of self
        """
        return self.point().isNear(pose.point(), distEps) and \
               nearAngle(self.theta, pose.pose().theta, angleEps)

    def diff(self, pose):
        """
        Return a pose that is the difference between self and pose (in
        x, y, z, and theta)
        """
        return Pose(self.x-pose.x,
                    self.y-pose.y,
                    self.z-pose.z,
                    fixAnglePlusMinusPi(self.theta-pose.theta))

    def distance(self, pose):
        """
        Return the distance between the x,y,z part of self and the x,y,z
        part of pose.
        """
        return self.point().distance(pose.point())

    def totalDist(self, pose, angleScale = 1):
        return self.distance(pose) + \
               abs(fixAnglePlusMinusPi(self.theta-pose.theta)) * angleScale

    def inverse(self):
        """
        Return a transformation matrix that is the inverse of the
        transform associated with this pose.
        """
        return super(Pose, self).inverse().pose()

    def xyztTuple(self):
        """
        Representation of pose as a tuple of values
        """
        return (self.x, self.y, self.z, self.theta)

    def corrupt(self, e, eAng = None):
        def corrupt(x, e):
            return x + random.uniform(-e, e)
        eAng = eAng or e
        return Pose(corrupt(self.x, e), corrupt(self.y, e), corrupt(self.z, e),
                    fixAnglePlusMinusPi(corrupt(self.theta, eAng)))

    def corruptGauss(self, mu, sigma, noZ = False):
        def corrupt(x):
            return x + random.gauss(mu, sigma)
        return Pose(corrupt(self.x), corrupt(self.y),
                    self.z if noZ else corrupt(self.z),
                    fixAnglePlusMinusPi(corrupt(self.theta)))

    def __repr__(self):
        if not self.reprString:
            # An attempt to make string equality useful
            self.reprString = 'Pose[' + prettyString(self.x) + ', ' +\
                              prettyString(self.y) + ', ' +\
                              prettyString(self.z) + ', ' +\
                              (prettyString(self.theta) \
                              if self.theta <= 6.283 else prettyString(0.0))\
                              + ']'
            #self.reprString = 'Pose'+ prettyString(self.xyztTuple())
        return self.reprString
    def shortStr(self, trim = False):
        return self.__repr__()
    def __eq__(self, other):
        return str(self) == str(other)
    def __hash__(self):
        return str(self).__hash__()
    __str__ = __repr__

class Point:
    """
    Represent a point with its x, y, z values
    """
    def __init__(self, x, y, z, w=1.0):
        self.x = x
        """x coordinate"""
        self.y = y
        """y coordinate"""
        self.z = z
        """z coordinate"""
        self.w = w
        """w coordinate"""

    def matrix(self):
        # recompute each time to allow changing coords... reconsider this later
        return numpy.array([self.x, self.y, self.z, self.w])

    def isNear(self, point, distEps):
        """
        Return true if the distance between self and point is less
        than distEps
        """
        return self.distance(point) < distEps

    def distance(self, point):
        """
        Euclidean distance between two points
        """
        dx = self.x - point.x
        dy = self.y - point.y
        dz = self.z - point.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def distanceXY(self, point):
        """
        Euclidean distance (squared) between two points
        """
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)

    def distanceSq(self, point):
        """
        Euclidean distance (squared) between two points
        """
        dx = self.x - point.x
        dy = self.y - point.y
        dz = self.z - point.z
        return dx*dx + dy*dy + dz*dz

    def distanceSqXY(self, point):
        """
        Euclidean distance (squared) between two points
        """
        dx = self.x - point.x
        dy = self.y - point.y
        return dx*dx + dy*dy

    def magnitude(self):
        """
        Magnitude of this point, interpreted as a vector in 3-space
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def xyzTuple(self):
        """
        Return tuple of x, y, z values
        """
        return (self.x, self.y, self.z)

    def pose(self, angle = 0.0):
        """
        Return a pose with the position of the point.
        """
        return Pose(self.x, self.y, self.z, angle)

    def point(self):
        """
        Return a point, that is, self.
        """
        return self

    def __repr__(self):
        if self.w == 1:
            return 'Point'+ prettyString(self.xyzTuple())
        if self.w == 0:
            return 'Delta'+ prettyString(self.xyzTuple())
        else:
            return 'PointW'+ prettyString(self.xyzTuple()+(self.w,))

    def shortStr(self, trim = False):
        return self.__repr__()

    def angleToXY(self, p):
        """
        Return angle in radians of vector from self to p (in the xy projection)
        """
        dx = p.x - self.x
        dy = p.y - self.y
        return math.atan2(dy, dx)

    def add(self, point):
        """
        Vector addition
        """
        return Point(self.x + point.x, self.y + point.y, self.z + point.z)
    def __add__(self, point):
        return self.add(point)
    def sub(self, point):
        """
        Vector subtraction
        """
        return Point(self.x - point.x, self.y - point.y, self.z - point.z)
    def __sub__(self, point):
        return self.sub(point)
    def scale(self, s):
        """
        Vector scaling
        """
        return Point(self.x*s, self.y*s, self.z*s)
    def __rmul__(self, s):
        return self.scale(s)
    def dot(self, p):
        """
        Dot product
        """
        return self.x*p.x + self.y*p.y + self.z*p.z

class LineXY:
    """
    Line in 2D space
    """
    def __init__(self, p1, p2):
        """
        Initialize with two points that are on the line.  Actually
        store a normal and an offset from the origin
        """
        self.theta = p1.angleToXY(p2)
        """normal angle"""
        self.nx = -math.sin(self.theta)
        """x component of normal vector"""
        self.ny = math.cos(self.theta)
        """y component of normal vector"""
        self.off = p1.x * self.nx + p1.y * self.ny
        """offset along normal"""

    def pointOnLine(self, p, eps):
        """
        Return true if p is within eps of the line
        """
        dist = abs(p.x*self.nx + p.y*self.ny - self.off)
        return dist < eps

    def __repr__(self):
        return 'LineXY'+ prettyString((self.nx, self.ny, self.off))
    def shortStr(self, trim = False):
        return self.__repr__()


class LineSeg(LineXY):
    """
    Line segment in 2D space
    """
    def __init__(self, p1, p2):
        """
        Initialize with two points that are on the line.  Store one of
        the points and the vector between them.
        """
        self.B = p1
        """One point"""
        self.C = p2
        """Other point"""
        self.M = p2 - p1
        """Vector from the stored point to the other point"""
        LineXY.__init__(self, p1, p2)
        """Initialize line attributes"""

    def closestPoint(self, p):
        """
        Return the point on the line that is closest to point p
        """
        t0 = self.M.dot(p - self.B) / self.M.dot(self.M)
        if t0 <= 0:
            return self.B
        elif t0 >= 1:
            return self.B + self.M
        else:
            return self.B + t0 * self.M

    def distToPoint(self, p):
        """
        Shortest distance between point p and this line
        """
        return p.distance(self.closestPoint(p))

    def __repr__(self):
        return 'LineSeg'+ prettyString((self.B, self.M))

#####################

def localToGlobal(pose, point):
    return pose.transformPoint(point)

def localPoseToGlobalPose(pose1, pose2):
    return pose1.compose(pose2)

# Given robot's pose in a global frame and a point in the global frame
# return coordinates of point in local frame
def globalToLocal(pose, point):
    return pose.inverse().transformPoint(point)

def globalPoseToLocalPose(pose1, pose2):
    return pose1.inverse().compose(pose2)

def sum(items):
    """
    Defined to work on items other than numbers, which is not true for
    the built-in sum.
    """
    if len(items) == 0:
        return 0
    else:
        result = items[0]
        for item in items[1:]:
            result += item
        return result

def smash(lists):
    return [item for sublist in lists for item in sublist]

def within(v1, v2, eps):
    """
    Return True if v1 is with eps of v2. All params are numbers
    """
    return abs(v1 - v2) < eps

def nearAngle(a1,a2,eps):
    """
    Return True if angle a1 is within epsilon of angle a2  Don't use
    within for this, because angles wrap around!
    """
    return abs(fixAnglePlusMinusPi(a1-a2)) < eps

def nearlyEqual(x,y):
    """
    Like within, but with the tolerance built in
    """
    return abs(x-y)<.0001

def fixAnglePlusMinusPi(a):
    """
    A is an angle in radians;  return an equivalent angle between plus
    and minus pi
    """
    pi2 = 2.0* math.pi
    while abs(a) > math.pi:
        if a > math.pi:
            a = a - pi2
        elif a < -math.pi:
            a = a + pi2
    return a

def fixAngle02Pi(a):
    """
    A is an angle in radians;  return an equivalent angle between 0 
    and 2 pi
    """
    pi2 = 2.0* math.pi
    while a < 0 or a > pi2: 
        if a < 0:
            a = a + pi2
        elif a > pi2:
            a = a - pi2
    return a

def reverseCopy(items):
    """
    Return a list that is a reversed copy of items
    """
    itemCopy = items[:]
    itemCopy.reverse()
    return itemCopy


def dotProd(a, b):
    """
    Return the dot product of two lists of numbers
    """
    return sum([ai*bi for (ai,bi) in zip(a,b)])

def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

def argmaxWithVal(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score and the score
    """
    best = l[0]; bestScore = f(best)
    for x in l:
        xScore = f(x)
        if xScore > bestScore:
            best, bestScore = x, xScore
    return (best, bestScore)

def argmaxIndex(l, f = lambda x: x):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the index of C{l} that has the highest score
    """
    best = 0; bestScore = f(l[best])
    for i in range(len(l)):
        xScore = f(l[i])
        if xScore > bestScore:
            best, bestScore = i, xScore
    return (best, bestScore)

def argmaxIndexWithTies(l, f = lambda x: x):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the index of C{l} that has the highest score
    """
    best = []; bestScore = f(l[0])
    for i in range(len(l)):
        xScore = f(l[i])
        if xScore > bestScore:
            best, bestScore = [i], xScore
        elif xScore == bestScore:
            best, bestScore = best + [i], xScore
    return (best, bestScore)


def randomMultinomial(dist):
    """
    @param dist: List of positive numbers summing to 1 representing a
    multinomial distribution over integers from 0 to C{len(dist)-1}.
    @returns: random draw from that distribution
    """
    r = random.random()
    for i in range(len(dist)):
        r = r - dist[i]
        if r < 0.0:
            return i
    return "weird"

def clip(v, vMin, vMax):
    """
    @param v: number
    @param vMin: number (may be None, if no limit)
    @param vMax: number greater than C{vMin} (may be None, if no limit)
    @returns: If C{vMin <= v <= vMax}, then return C{v}; if C{v <
    vMin} return C{vMin}; else return C{vMax}
    """
    try:
        return [clip(x, vMin, vMax) for x in v]
    except TypeError:
        if vMin == None:
            if vMax == None:
                return v
            else:
                return min(v, vMax)
        else:
            if vMax == None:
                return max(v, vMin)
            else:
                return max(min(v, vMax), vMin)

def flatten(M):
    """
    basically a nice wrapper around reshape
    @param M: matrix
    @returns v: flattened matrix into a vector
    """
    return np.reshape(M, (M.shape[0]*M.shape[1]))
def sign(x):
    """
    Return 1, 0, or -1 depending on the sign of x
    """
    if x > 0.0:
        return 1
    elif x == 0.0:
        return 0
    else:
        return -1

def make2DArray(dim1, dim2, initValue):
    """
    Return a list of lists representing a 2D array with dimensions
    dim1 and dim2, filled with initialValue
    """
    result = []
    for i in range(dim1):
        result = result + [makeVector(dim2, initValue)]
    return result

def make2DArrayFill(dim1, dim2, initFun):
    """
    Return a list of lists representing a 2D array with dimensions
    dim1 and dim2, filled by calling initFun with every pair of
    indices 
    """
    result = []
    for i in range(dim1):
        result = result + [makeVectorFill(dim2, lambda j: initFun(i, j))]
    return result

def make3DArray(dim1, dim2, dim3, initValue):
    """
    Return a list of lists of lists representing a 3D array with dimensions
    dim1, dim2, and dim3 filled with initialValue
    """
    result = []
    for i in range(dim1):
        result = result + [make2DArray(dim2, dim3, initValue)]
    return result

def mapArray3D(array, f):
    """
    Map a function over the whole array.  Side effects the array.  No
    return value.
    """
    for i in range(len(array)):
        for j in range(len(array[0])):
            for k in range(len(array[0][0])):
                array[i][j][k] = f(array[i][j][k])

def makeVector(dim, initValue):
    """
    Return a list of dim copies of initValue
    """
    return [initValue]*dim

def makeVectorFill(dim, initFun):
    """
    Return a list resulting from applying initFun to values from 0 to
    dim-1
    """
    return [initFun(i) for i in range(dim)]

def prettyString(struct):
    """
    Make nicer looking strings for printing, mostly by truncating
    floats
    """
    if type(struct) == list:
        return '[' + ', '.join([prettyString(item) for item in struct]) + ']'
    elif type(struct) == tuple:
        return '(' + ', '.join([prettyString(item) for item in struct]) + ')'
    elif type(struct) == dict:
        return '{' + ', '.join([str(item) + ':' +  prettyString(struct[item]) \
                                             for item in struct]) + '}'
    elif type(struct) == float or type(struct) == numpy.float64:
        struct = round(struct, 3)
        if struct == 0: struct = 0      #  catch stupid -0.0
        return "%5.3f" % struct
    else:
        return str(struct)

def swapRange(x, y):
    if x < y:
        return range(x, y)
    if x > y:
        r = range(y, x)
        r.reverse()
        return r
    return [x]

def avg(a, b):
    if type(a) in (types.TupleType, types.ListType) and \
            type(b) in (types.TupleType, types.ListType) and \
            len(a) == len(b):
        return tuple([avg(a[i], b[i]) for i in range(len(a))])
    else:
        return (a + b)/2.0
def recoverPath(volume, start, end):
    if not volume:
        return None
    p = []
    current = start
    while current != end:
        p.append(current)
        successors = [[current[0] + i, current[1] + j] for (i,j) in [(1,0), (0, 1), (-1, 0), (0, -1)]]
        for v in volume:
            if list(v) in successors and not list(v) in p:
                current = list(v)
                continue
    p.append(end)
    return p
  
class SymbolGenerator:
    """
    Generate new symbols guaranteed to be different from one another
    Optionally, supply a prefix for mnemonic purposes
    Call gensym("foo") to get a symbol like 'foo37'
    """
    def __init__(self):
        self.count = 0
    def gensym(self, prefix = 'i'):
        self.count += 1
        return prefix + '_' + str(self.count)
    
gensym = SymbolGenerator().gensym
"""Call this function to get a new symbol"""

def logGaussian(x, mu, sigma):
    """
    Log of the value of the gaussian distribution with mean mu and
    stdev sigma at value x
    """
    return -((x-mu)**2 / (2*sigma**2)) - math.log(sigma*math.sqrt(2*math.pi))

def gaussian(x, mu, sigma):
    """
    Value of the gaussian distribution with mean mu and
    stdev sigma at value x
    """
    return math.exp(-((x-mu)**2 / (2*sigma**2))) /(sigma*math.sqrt(2*math.pi))  

def lineIndices((i0, j0), (i1, j1)):
    """
    Takes two cells in the grid (each described by a pair of integer
    indices), and returns a list of the cells in the grid that are on the
    line segment between the cells.
    """
    ans = [(i0,j0)]
    di = i1 - i0
    dj = j1 - j0
    t = 0.5
    if abs(di) > abs(dj):               # slope < 1
        m = float(dj) / float(di)       # compute slope
        t += j0
        if di < 0: di = -1
        else: di = 1
        m *= di
        while (i0 != i1):
            i0 += di
            t += m
            ans.append((i0, int(t)))
    else:
        if dj != 0:                     # slope >= 1
            m = float(di) / float(dj)   # compute slope
            t += i0
            if dj < 0: dj = -1
            else: dj = 1
            m *= dj
            while j0 != j1:
                j0 += dj
                t += m
                ans.append((int(t), j0))
    return ans

def angleDiff(x, y):
    twoPi = 2*math.pi
    z = (x - y)%twoPi
    if z > math.pi:
        return z - twoPi
    else:
        return z

def inRange(v, r):
    return r[0] <= v <= r[1]

def rangeOverlap(r1, r2):
    return r2[0] <= r1[1] and r1[0] <= r2[1]

def rangeIntersect(r1, r2):
    return (max(r1[0], r2[0]), min(r1[1], r2[1]))

def average(stuff):
    return (1./float(len(stuff)))*sum(stuff)

def tuplify(x):
    if isIterable(x):
        return tuple([tuplify(y) for y in x])
    else:
        return x

def squash(listOfLists):
    return reduce(operator.add, listOfLists)

# Average two angles
def angleAverage(th1, th2, alpha):
    return math.atan2(alpha * math.sin(th1) + (1 - alpha) * math.sin(th2),
                      alpha * math.cos(th1) + (1 - alpha) * math.cos(th2))
    
def floatRange(lo, hi, stepsize):
    """
    @returns: a list of numbers, starting with C{lo}, and increasing
    by C{stepsize} each time, until C{hi} is equaled or exceeded.

    C{lo} must be less than C{hi}; C{stepsize} must be greater than 0.
    """
    if stepsize == 0:
       print 'Stepsize is 0 in floatRange'
    result = []
    v = lo
    while v <= hi:
        result.append(v)
        v += stepsize
    return result

def euclideanDistance(x, y):
    return math.sqrt(sum([(xi - yi)**2 for (xi, yi) in zip(x, y)]))

def pop(x):
    if isinstance(x, list):
        if len(x) > 0:
            return x.pop(0)
        else:
            return None
    else:
        try:
            return x.next()
        except StopIteration:
            return None

def isIterable(x):
    if type(x) in (str, unicode):
        return False
    try:
        x_iter = iter(x)
        return True
    except:
        return False

def tangentSpaceAdd(a, b):
    res = a + b
    for i in range(3, len(res), 4):
        res[i, 0] = fixAnglePlusMinusPi(res[i, 0])
    return res

def scalarMult(l, c):
    return type(l)([i*c for i in l])

def componentAdd(a, b):
    return type(a)([i + j for (i, j) in zip(a, b)])

def componentSubtract(a, b):
    return componentAdd(a, [-1*i for i in b])        
