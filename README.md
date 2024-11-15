# <div align="center">YOLO11-psen</div>

We used the SHWD dataset and used Real-ESRGAN for data enhancement

Datasets: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset

Real-ESRGAN： https://github.com/xinntao/Real-ESRGAN



<details open>
<summary>Processing the dataset</summary>

Use the `Real-ESRGAN` model to process the dataset. We use its default pre-trained weights [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth).
The command is:
```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```
</details>

Among them, `inputs` is the folder where the images to be processed are located. You can also specify the image files you want to process. We also specify the face enhancement option because processing the SHWD dataset contains many face images.


<details open>
<summary>YOLO11-psen modification instructions</summary>


The relevant configuration file is in [ultralytics/cfg/models/11-T/yolo11n_ECA_SPPELAN_C3K2PConv.yaml](https://github.com/lixin123-lixin-123/yolo11n-PSEN/blob/master/ultralytics/cfg/models/11-T/yolo11n_ECA_SPPELAN_C3K2PConv.yaml) 。


Compared with the original `yolo11n.yaml`, we replaced the `C3K2` module with our modified `C3K2_PConv` module, which can be compared and analyzed in the 2nd, 4th, 6th, 8th, 13th, 16th, 19th, and 22nd layers of the yaml file; 

replaced the `SPPF` module with our `SPPELAN` module in the 9th layer of the model;

replaced the `C2PSA` module with our `ECA` module in the 10th layer of the model, changed the number of times the layer is applied to 1, and changed the number of channels to adaptive inheritance. The number of `ECA` channels is usually automatically inherited from the output channel number of the previous layer, so it does not need to be explicitly specified. It can be automatically adapted through the previous layer's output, making the configuration file more flexible and versatile. For example, if the number of output channels of the previous layer of `ECA` is 1024, then `ECA` will use this number of channels. 


The relevant implementation files are in [ultralytics/nn/models](https://github.com/lixin123-lixin-123/yolo11n-PSEN/tree/master/ultralytics/nn/modules), namely [PConv.py](https://github.com/lixin123-lixin-123/yolo11n-PSEN/blob/master/ultralytics/nn/modules/PConv.py), [SPPELAN.py](https://github.com/lixin123-lixin-123/yolo11n-PSEN/blob/master/ultralytics/nn/modules/SPPELAN.py), and [ECA.py](https://github.com/lixin123-lixin-123/yolo11n-PSEN/blob/master/ultralytics/nn/modules/ECA.py). Then they are declared and introduced in [ultralytics/nn/task.py](https://github.com/lixin123-lixin-123/yolo11n-PSEN/blob/master/ultralytics/nn/tasks.py). The specific modified code is in lines 85-87 and 969-1041.

Finally, we perform a weighted fusion of `NWD` and `CIoU`. The implementation code is in [ultralytics/utils/loss.py](https://github.com/lixin123-lixin-123/yolo11n-PSEN/blob/master/ultralytics/utils/loss.py). First, the definition of `NWD` is introduced at the end of the file, then it is declared and used in lines 113-116, and the weighted formula is used for fusion.

<details open>
<summary>Environment Configuration</summary>

pip installs the ultralytics package with all [requirements.txt](https://github.com/lixin123-lixin-123/yolo11n-PSEN/blob/master/requirements.txt), **Python=3.9**, and **PyTorch==2.4.0**.

```bash
pip install ultralytics
```
</details>



<details open>
<summary>How to use</summary>

```bash
yolo detect train data=aqm-coco.yaml model=./ultralytics/cfg/models/11-T/yolo11n_ECA_SPPELAN_C3K2PConv.yaml  epochs=300 batch=256 imgsz=640 device=0,1,2,3
```

For specific commands, please refer to the training commands of  `yolov8` `yolov10` `yolo11`


