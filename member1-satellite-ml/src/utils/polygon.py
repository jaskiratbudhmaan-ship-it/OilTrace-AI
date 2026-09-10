import cv2, numpy as np

def clean_binary_mask(mask,min_component_area=20):
    mask=(mask>0).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    out=np.zeros_like(mask)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=min_component_area:
            out[labels==i]=1
    return out

def mask_to_polygons(mask,min_area=20):
    mask=clean_binary_mask(mask,min_area)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    polys=[]
    for c in contours:
        if cv2.contourArea(c)<min_area: continue
        a=cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        polys.append([[int(p[0][0]),int(p[0][1])] for p in a])
    return polys

def centroid_from_mask(mask):
    m=cv2.moments((mask>0).astype(np.uint8))
    if m["m00"]==0: return None
    return [float(m["m10"]/m["m00"]),float(m["m01"]/m["m00"])]
