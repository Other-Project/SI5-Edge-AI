import blobconverter
import os


def export_to_oak_blob(onnx_path, datatype="FP16"):

    blob_path = blobconverter.from_onnx(
        model=onnx_path,
        data_type=datatype,
        shaves=8,
        use_cache=True,
        optimizer_params=[
            "--scale_values=[255,255,255]"
        ],
        output_dir="../models/"
    )
    print(f"Blob créé : {blob_path}")

if __name__ == "__main__":
    for model in os.listdir('../models'):
        if "fp32" in model:
            export_to_oak_blob(onnx_path=os.path.join('../models/', model), datatype="FP32")
        elif "int8" in model:
            continue
        else:
            export_to_oak_blob(onnx_path=os.path.join('../models/', model))