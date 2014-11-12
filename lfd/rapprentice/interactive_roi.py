import cv2
import numpy as np

def get_polyline(image,window_name):
    cv2.namedWindow(window_name)
    class GetPoly:
        xys = []        
        done = False
        def callback(self,event, x, y, flags, param):
            if self.done == True:
                pass
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.xys.append((x,y))
            elif event == cv2.EVENT_MBUTTONDOWN:
                self.done = True
    gp = GetPoly()
    cv2.setMouseCallback(window_name,gp.callback)
    print "press middle mouse button or 'c' key to complete the polygon"
    while not gp.done:
        im_copy = image.copy()
        for (x,y) in gp.xys:
            cv2.circle(im_copy,(x,y),2,(0,255,0))
        if len(gp.xys) > 1 and not gp.done:
            cv2.polylines(im_copy,[np.array(gp.xys).astype('int32')],False,(0,255,0),1)
        cv2.imshow(window_name,im_copy)
        key = cv2.waitKey(50)
        if key == ord('c'): gp.done = True
    #cv2.destroyWindow(window_name)
    return gp.xys

def get_polygon_and_prompt(image, window_name):
    im_copy = image.copy()
    xys = get_polyline(image,window_name)
    assert len(xys)>1
    cv2.polylines(im_copy,[np.array(xys+xys[0:1]).astype('int32')],False,(0,0,255),1)
    cv2.imshow(window_name,im_copy)

    print "press 'a' to accept, 'r' to reject"    
    while True:
        key = cv2.waitKey(0)
        if key == ord('a'): return np.array(xys)
        else: exit(1)

    return np.array(xys)


def mask_from_poly(xys,shape=(480,640)):
    img = np.zeros(shape,dtype='uint8')
    xys = np.array(xys).astype('int32')
    cv2.fillConvexPoly(img,xys,(255,0,0))
    return img.copy()
