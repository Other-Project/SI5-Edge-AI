from ultralytics import YOLO
import ultralytics
from tqdm import tqdm
from ultralytics import settings


model = YOLO('yolo11.yaml')

import os
import shutil

def organize_onnx_files(base_runs_dir="./runs/segment", destination_dir="../models/"):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Folder created: {destination_dir}")

    for folder_name in os.listdir(base_runs_dir):
        onnx_path = os.path.join(base_runs_dir, folder_name, "weights", "yolo11n_seg_prune_0_2.onnx")

        if os.path.exists(onnx_path):
            new_name = f"{folder_name}.onnx"
            final_name = new_name.replace('.', '_').replace(',', '_')

            final_destination = os.path.join(destination_dir, final_name)

            shutil.copy2(onnx_path, final_destination)
            print(f"Copied and renamed: {new_name}")

def prunetrain(model, train_epochs, prune_epochs=0, quick_pruning=True, prune_ratio=0.5,
               prune_iterative_steps=1, data='coco.yaml', name='yolo11', imgsz=640, 
               batch=8, device=[0], sparse_training=False):
    if not quick_pruning:
        assert train_epochs > 0 and prune_epochs > 0, "Quick Pruning is not set. prune epochs must > 0."
        model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch, device=device, name=name, prune=False,
                    sparse_training=sparse_training)
        return model.train(data=data, epochs=prune_epochs, imgsz=imgsz, batch=batch, device=device, name=name, prune=True,
                           prune_ratio=prune_ratio, prune_iterative_steps=prune_iterative_steps)
    else:
        return model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch, device=device, 
                           name=name, prune=True, prune_ratio=prune_ratio, prune_iterative_steps=prune_iterative_steps)



if __name__ == '__main__':
    for k in tqdm([0.05, 0.1, 0.15, 0.25, 0.5, 0.75]):
        model_name = f"yolo11n_seg_prune_{k}"
        # Initialize your model here or inside the function
        model = YOLO('yolo11n-seg.pt')

        # Normal Pruning
        prunetrain(
            model = model,
            quick_pruning=True,
            data='voc500.yaml',
            train_epochs=25,
            prune_epochs=5,
            imgsz=640,
            batch=5,
            device=[0],
            name=model_name,
            prune_ratio=k,
            prune_iterative_steps=1,
            sparse_training=False
        )

        model = YOLO("./runs/segment/"+model_name+'/weights/best.pt')
        model.export(format='onnx')
        organize_onnx_files()