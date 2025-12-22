
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval_bbox(gt_json, pred_json):
    cocoGt = COCO(gt_json)
    cocoDt = cocoGt.loadRes(pred_json)
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    eval_bbox("data/test/test.json", "experiments/point_rend/predictions_test_499.json")
