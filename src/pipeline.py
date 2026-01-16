import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
import time

from .helpers.utils import get_dataset_paths

# Benchmark settings
MAX_IMAGES = 100
DATA_DIR, LABELS_FILE = get_dataset_paths("coco_person")

class BenchmarkPipeline:
    def __init__(self, data_path=DATA_DIR,labels_path=LABELS_FILE):
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

    def bench(self, backend_class, models, output_dir, result_path = 'benchmark.csv', max_images=MAX_IMAGES, backend_kwargs=None):
        if backend_kwargs is None:
            backend_kwargs = {}
            
        self.result_path = result_path
        results = {}
        model_list = list(models.items()) # On transforme en liste pour savoir si on est au dernier
        
        for i, (model_name, model_path) in enumerate(model_list):
            try:
                print(f"\n" + "="*50)
                print(f"--- Benchmarking Model {i+1}/{len(model_list)}: {model_name} ---")
                print("="*50)

                backend = backend_class(model_path, **backend_kwargs)
                stats = self.run(backend, max_images=max_images)
                results[model_name] = stats
                backend.close()
                
                print(f"\n[INFO] Benchmark terminé pour {model_name}.")
                print("Attente de 10 secondes pour laisser le matériel refroidir...")
                time.sleep(10)

            except Exception as e:
                print(f"Error processing model {model_name}: {e}")

        self.export_to_csv(results, output_dir)
        return results


    def run(self, backend, max_images=None):
        coco_results = []
        times = []
        
        img_list = self.images[:max_images] if max_images else self.images
        
        print(f"Starting benchmark on {len(img_list)} images with {backend.__class__.__name__}...")
        
        for img_name in tqdm(img_list):
            path = os.path.join(self.data_path, img_name)
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None: 
                continue
            
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
    
    def export_to_csv(self, stats, output_dir):
        if not stats:
            return
        
        csv_path = os.path.join(output_dir, self.result_path)
        first_model_stats = next(iter(stats.values()))
        
        fieldnames = ['Model'] + list(first_model_stats.keys())

        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            import csv
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            
            for model_name, model_stats in stats.items():
                row = {'Model': model_name}
                row.update(model_stats)
                writer.writerow(row)

# Global benchmark instance
benchmark = BenchmarkPipeline()