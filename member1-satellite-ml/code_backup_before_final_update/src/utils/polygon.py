import cv2, numpy as np

def mask_to_polygons(mask):
    mask=(mask>0).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    polys=[]
    for c in contours:
        if cv2.contourArea(c)<10:
            continue
        approx=cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        polys.append([[int(p[0][0]),int(p[0][1])] for p in approx])
    return polys

def largest_centroid(mask):
    m=cv2.moments((mask>0).astype(np.uint8))
    if m["m00"]==0: return None
    return [m["m10"]/m["m00"], m["m01"]/m["m00"]]
