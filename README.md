# Edge AI OpenVINO PIFu Convert

## Requirements

Please Execute the Follow Commands to Fullfill the Requirements

```
sudo apt install python3.9
pip3 install rembg
pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install scikit-image
pip3 install opencv-python
pip3 install numpy
pip3 install openvino
```

## Inference

Please Execute the Following Command to Generate 3D Model Based on the "source.png" image.

```
python inference_image.py
```

The Generated 3D Model will be saved in the file "model.obj".
