import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

class BenchmarkPipeline:
    def __init__(self, data_path, labels_path):
        self.data_path = data_path
        self.coco_gt = COCO(labels_path)
        
        self.person_cat_id = 1
        cats = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        
        for cat in cats:
            if 'person' in cat['name'].lower():
                self.person_cat_id = cat['id']
                break
        
        self.coco_imgs = self.coco_gt.loadImgs(self.coco_gt.getImgIds())
        self.filename_to_id = {img['file_name']: img['id'] for img in self.coco_imgs}
        
        exts = ('.jpg', '.png', '.jpeg', '.bmp')
        self.images = [f for f in os.listdir(data_path) if f.lower().endswith(exts)]

        self.images = [f for f in self.images if f in self.filename_to_id]

    def run(self, backend, max_images=None):
        coco_results = []
        times = []
        
        img_list = self.images[:max_images] if max_images else self.images
        
        print(f"Starting benchmark on {len(img_list)} images with {backend.__class__.__name__}...")
        
        for img_name in tqdm(img_list):
            path = os.path.join(self.data_path, img_name)
            img = cv2.imread(path)
            if img is None: continue
            
            image_id = self.filename_to_id[img_name]

            decoder, t_ms = backend.predict(img)
            times.append(t_ms)
            
            if len(decoder.scores) > 0:
                for i, score in enumerate(decoder.scores):
                    
                    if int(decoder.class_ids[i]) != 0:
                        continue

                    binary_mask = np.asfortranarray(decoder.mask_maps[i].astype(np.uint8))
                    rle = mask_utils.encode(binary_mask)
                    
                    x1, y1, x2, y2 = decoder.boxes[i]
                    bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
                    
                    res = {
                        "image_id": image_id,
                        "category_id": self.person_cat_id,
                        "bbox": bbox,
                        "segmentation": rle,
                        "score": float(score)
                    }
                    coco_results.append(res)
                    
        stats = {
            "mAP_box": 0.0, 
            "mAP_mask": 0.0, 
            "time_avg": 0.0,
            "time_min": 0.0,
            "time_max": 0.0,
            "time_std": 0.0
        }
        
        if len(times) > 0:
            stats["time_avg"] = np.mean(times)
            stats["time_min"] = np.min(times)
            stats["time_max"] = np.max(times)
            stats["time_std"] = np.std(times)
            
        if len(coco_results) > 0:
            coco_dt = self.coco_gt.loadRes(coco_results)
            img_ids = [self.filename_to_id[f] for f in img_list]
            
            print("\n--- Evaluation BBOX (Person Only) ---")
            eval_box = COCOeval(self.coco_gt, coco_dt, 'bbox')
            eval_box.params.imgIds = img_ids

            eval_box.params.catIds = [self.person_cat_id]
            eval_box.evaluate()
            eval_box.accumulate()
            eval_box.summarize()
            stats["mAP_box"] = eval_box.stats[0]
            
            if any("segmentation" in r for r in coco_results):
                print("\n--- Evaluation MASK (Person Only) ---")
                eval_seg = COCOeval(self.coco_gt, coco_dt, 'segm')
                eval_seg.params.imgIds = img_ids
                eval_seg.params.catIds = [self.person_cat_id]
                eval_seg.evaluate()
                eval_seg.accumulate()
                eval_seg.summarize()
                stats["mAP_mask"] = eval_seg.stats[0]

        return stats