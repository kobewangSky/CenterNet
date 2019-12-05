import os
import torch


os.environ["CUDA_VISIBLE_DEVICE"] = "1"
model = torch.load("./exp/ctdet/ctdet_coco_hg.pth")
del model['iteration']
del model['scheduler']
del model['optimizer']
del model['model']['module.roi_heads.box.predictor.cls_score.weight']
del model['model']['module.roi_heads.box.predictor.cls_score.bias']
del model['model']['module.roi_heads.box.predictor.bbox_pred.weight']
del model['model']['module.roi_heads.box.predictor.bbox_pred.bias']
del model['model']['module.roi_heads.mask.predictor.mask_fcn_logits.weight']
del model['model']['module.roi_heads.mask.predictor.mask_fcn_logits.bias']
del model['model']['module.rpn.head.cls_logits.weight']
del model['model']['module.rpn.head.cls_logits.bias']
del model['model']['module.rpn.head.bbox_pred.weight']
del model['model']['module.rpn.head.bbox_pred.bias']
torch.save(model, "./ACRV/CoCo_Virtual_30Class/model_0000000_.pth")