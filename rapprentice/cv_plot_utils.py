import cv2, numpy as np

class Colors:
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)



class ClickGetter:
    xy = None
    done = False
    def callback(self,event, x, y, _flags, _param):
        if self.done:
            return
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.xy = (x,y)
            self.done = True
def get_click(windowname, img):
    cg = ClickGetter()
    cv2.setMouseCallback(windowname, cg.callback)
    while not cg.done:
        cv2.imshow(windowname, img)
        cv2.waitKey(10)
    return cg.xy

def draw_img(img, colormap = None, min_size = 1):
    if img.dtype == np.bool:
        img = img.astype('uint8')
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = img.astype("float32")
        minval = img.min()
        maxval = img.max()
        img = (img-minval)/(maxval-minval)
        img = (img*256).astype('uint8')
    if img.shape[0] < min_size:
        ratio = int(np.ceil(float(min_size)/img.shape[0]))
        img = cv2.resize(img, (img.shape[0]*ratio, img.shape[1]*ratio))
        
    if colormap is not None:
        img = colormap[img]
        
    cv2.imshow("draw_img", img)
    cv2.waitKey()
    cv2.destroyWindow("draw_img")


def tile_images(imgs, nrows, ncols, row_titles = None, col_titles = None, max_width = 1000):
    assert nrows*ncols >= len(imgs)
    if nrows*ncols > len(imgs):
        imgs = [img for img in imgs]
        imgs.extend([np.zeros_like(imgs[0]) for _ in xrange(nrows*ncols - len(imgs))])
    full_width = imgs[0].shape[1]*ncols
    if full_width > max_width:
        ratio = float(max_width)/ full_width
        for i in xrange(len(imgs)): 
            imgs[i] = cv2.resize(imgs[i], (int(imgs[i].shape[1]*ratio), int(imgs[i].shape[0]*ratio)))
    
    if col_titles is not None: raise NotImplementedError
    imgrows = []
    for irow in xrange(nrows):
        rowimgs = imgs[irow*ncols:(irow+1)*ncols]
        if row_titles is not None:
            rowimgs[0] = rowimgs[0].copy()
            cv2.putText(rowimgs[0], row_titles[irow], (10,10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), thickness = 1)            
        imgrows.append(np.concatenate(rowimgs,1))
    bigimg = np.concatenate(imgrows, 0) 
    return bigimg

def label2rgb(labels):
    max_label = labels.max()
    rgbs = (np.random.rand(max_label+1,3)*256).astype('uint8')
    return rgbs[labels]

def inttuple(x,y):
    return int(np.round(x)), int(np.round(y))


def circle_with_ori(img, x, y, theta):    
    cv2.circle(img, inttuple(x, y), 17, (0,255,255), 3)
    cv2.line(img, inttuple(x,y), inttuple(x+17*np.cos(theta), y+17*np.sin(theta)), (0,255,255), 3)



CM_JET = np.array([
       [  0,   0, 128],
       [  0,   0, 132],
       [  0,   0, 137],
       [  0,   0, 141],
       [  0,   0, 146],
       [  0,   0, 150],
       [  0,   0, 155],
       [  0,   0, 159],
       [  0,   0, 164],
       [  0,   0, 169],
       [  0,   0, 173],
       [  0,   0, 178],
       [  0,   0, 182],
       [  0,   0, 187],
       [  0,   0, 191],
       [  0,   0, 196],
       [  0,   0, 201],
       [  0,   0, 205],
       [  0,   0, 210],
       [  0,   0, 214],
       [  0,   0, 219],
       [  0,   0, 223],
       [  0,   0, 228],
       [  0,   0, 232],
       [  0,   0, 237],
       [  0,   0, 242],
       [  0,   0, 246],
       [  0,   0, 251],
       [  0,   0, 255],
       [  0,   0,   0],
       [  0,   0,   0],
       [  0,   0,   0],
       [  0,   0,   0],
       [  0,   4,   0],
       [  0,   8,   0],
       [  0,  12,   0],
       [  0,  16,   0],
       [  0,  20,   0],
       [  0,  24,   0],
       [  0,  28,   0],
       [  0,  32,   0],
       [  0,  36,   0],
       [  0,  40,   0],
       [  0,  44,   0],
       [  0,  48,   0],
       [  0,  52,   0],
       [  0,  56,   0],
       [  0,  60,   0],
       [  0,  64,   0],
       [  0,  68,   0],
       [  0,  72,   0],
       [  0,  76,   0],
       [  0,  80,   0],
       [  0,  84,   0],
       [  0,  88,   0],
       [  0,  92,   0],
       [  0,  96,   0],
       [  0, 100,   0],
       [  0, 104,   0],
       [  0, 108,   0],
       [  0, 112,   0],
       [  0, 116,   0],
       [  0, 120,   0],
       [  0, 124,   0],
       [  0, 129,   0],
       [  0, 133,   0],
       [  0, 137,   0],
       [  0, 141,   0],
       [  0, 145,   0],
       [  0, 149,   0],
       [  0, 153,   0],
       [  0, 157,   0],
       [  0, 161,   0],
       [  0, 165,   0],
       [  0, 169,   0],
       [  0, 173,   0],
       [  0, 177,   0],
       [  0, 181,   0],
       [  0, 185,   0],
       [  0, 189,   0],
       [  0, 193,   0],
       [  0, 197,   0],
       [  0, 201,   0],
       [  0, 205,   0],
       [  0, 209,   0],
       [  0, 213,   0],
       [  0, 217,   0],
       [  0, 221, 255],
       [  0, 225, 251],
       [  0, 229, 248],
       [  2, 233, 245],
       [  5, 237, 242],
       [  8, 241, 238],
       [ 12, 245, 235],
       [ 15, 249, 232],
       [ 18, 253, 229],
       [ 21,   0, 225],
       [ 25,   0, 222],
       [ 28,   0, 219],
       [ 31,   0, 216],
       [ 34,   0, 212],
       [ 38,   0, 209],
       [ 41,   0, 206],
       [ 44,   0, 203],
       [ 47,   0, 199],
       [ 51,   0, 196],
       [ 54,   0, 193],
       [ 57,   0, 190],
       [ 60,   0, 187],
       [ 63,   0, 183],
       [ 67,   0, 180],
       [ 70,   0, 177],
       [ 73,   0, 174],
       [ 76,   0, 170],
       [ 80,   0, 167],
       [ 83,   0, 164],
       [ 86,   0, 161],
       [ 89,   0, 157],
       [ 93,   0, 154],
       [ 96,   0, 151],
       [ 99,   0, 148],
       [102,   0, 144],
       [106,   0, 141],
       [109,   0, 138],
       [112,   0, 135],
       [115,   0, 131],
       [119,   0, 128],
       [122,   0, 125],
       [125,   0, 122],
       [128,   0, 119],
       [131,   0, 115],
       [135,   0, 112],
       [138,   0, 109],
       [141,   0, 106],
       [144,   0, 102],
       [148,   0,  99],
       [151,   0,  96],
       [154,   0,  93],
       [157,   0,  89],
       [161,   0,  86],
       [164,   0,  83],
       [167,   0,  80],
       [170,   0,  76],
       [174,   0,  73],
       [177,   0,  70],
       [180,   0,  67],
       [183,   0,  63],
       [187,   0,  60],
       [190,   0,  57],
       [193,   0,  54],
       [196,   0,  51],
       [199,   0,  47],
       [203,   0,  44],
       [206,   0,  41],
       [209,   0,  38],
       [212,   0,  34],
       [216,   0,  31],
       [219,   0,  28],
       [222,   0,  25],
       [225,   0,  21],
       [229,   0,  18],
       [232,   0,  15],
       [235,   0,  12],
       [238,   0,   8],
       [242, 253,   5],
       [245, 249,   2],
       [248, 245,   0],
       [251, 241,   0],
       [255, 238,   0],
       [  0, 234,   0],
       [  0, 230,   0],
       [  0, 226,   0],
       [  0, 223,   0],
       [  0, 219,   0],
       [  0, 215,   0],
       [  0, 212,   0],
       [  0, 208,   0],
       [  0, 204,   0],
       [  0, 200,   0],
       [  0, 197,   0],
       [  0, 193,   0],
       [  0, 189,   0],
       [  0, 186,   0],
       [  0, 182,   0],
       [  0, 178,   0],
       [  0, 174,   0],
       [  0, 171,   0],
       [  0, 167,   0],
       [  0, 163,   0],
       [  0, 160,   0],
       [  0, 156,   0],
       [  0, 152,   0],
       [  0, 148,   0],
       [  0, 145,   0],
       [  0, 141,   0],
       [  0, 137,   0],
       [  0, 134,   0],
       [  0, 130,   0],
       [  0, 126,   0],
       [  0, 122,   0],
       [  0, 119,   0],
       [  0, 115,   0],
       [  0, 111,   0],
       [  0, 108,   0],
       [  0, 104,   0],
       [  0, 100,   0],
       [  0,  96,   0],
       [  0,  93,   0],
       [  0,  89,   0],
       [  0,  85,   0],
       [  0,  81,   0],
       [  0,  78,   0],
       [  0,  74,   0],
       [  0,  70,   0],
       [  0,  67,   0],
       [  0,  63,   0],
       [  0,  59,   0],
       [  0,  55,   0],
       [  0,  52,   0],
       [  0,  48,   0],
       [  0,  44,   0],
       [  0,  41,   0],
       [  0,  37,   0],
       [  0,  33,   0],
       [  0,  29,   0],
       [  0,  26,   0],
       [  0,  22,   0],
       [255,  18,   0],
       [251,  15,   0],
       [246,  11,   0],
       [242,   7,   0],
       [237,   3,   0],
       [232,   0,   0],
       [228,   0,   0],
       [223,   0,   0],
       [219,   0,   0],
       [214,   0,   0],
       [210,   0,   0],
       [205,   0,   0],
       [201,   0,   0],
       [196,   0,   0],
       [191,   0,   0],
       [187,   0,   0],
       [182,   0,   0],
       [178,   0,   0],
       [173,   0,   0],
       [169,   0,   0],
       [164,   0,   0],
       [159,   0,   0],
       [155,   0,   0],
       [150,   0,   0],
       [146,   0,   0],
       [141,   0,   0],
       [137,   0,   0],
       [132,   0,   0],
       [128,   0,   0]], dtype=np.uint8)

