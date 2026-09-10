import torch

def segmentation_metrics(logits, targets, threshold=0.5, eps=1e-7):
    probs=torch.sigmoid(logits)
    preds=(probs>=threshold).float()
    targets=(targets>=0.5).float()

    tp=(preds*targets).sum().item()
    fp=(preds*(1-targets)).sum().item()
    fn=((1-preds)*targets).sum().item()

    precision=tp/(tp+fp+eps)
    recall=tp/(tp+fn+eps)
    iou=tp/(tp+fp+fn+eps)
    dice=(2*tp)/(2*tp+fp+fn+eps)
    f1=2*precision*recall/(precision+recall+eps)

    return {"precision":precision,"recall":recall,"iou":iou,"dice":dice,"f1":f1}
