import torch

def batch_confusion(logits,targets,threshold=0.5):
    preds=(torch.sigmoid(logits)>=threshold).float()
    targets=(targets>=0.5).float()
    tp=(preds*targets).sum().item()
    fp=(preds*(1-targets)).sum().item()
    fn=((1-preds)*targets).sum().item()
    tn=((1-preds)*(1-targets)).sum().item()
    return tp,fp,fn,tn

def metrics_from_counts(tp,fp,fn,tn,eps=1e-7):
    p=tp/(tp+fp+eps); r=tp/(tp+fn+eps)
    iou=tp/(tp+fp+fn+eps)
    dice=(2*tp)/(2*tp+fp+fn+eps)
    f1=2*p*r/(p+r+eps)
    acc=(tp+tn)/(tp+fp+fn+tn+eps)
    return {"oil_iou":iou,"dice":dice,"precision":p,"recall":r,"f1":f1,"pixel_accuracy":acc}
