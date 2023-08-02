<p align="center">

<h1 align="center">EC-classifier-training-framework</h1>
<p align="center">
Hongqing Liu, 

</p>
    
## Run
Here you can run EC-Fusion-RGBIR yourself using terminal command.

```bash
python -W ignore run.py config/RGB_image.yaml
```

### configuration file
Here you can change configuration file to load different datasets and using different network

```bash
dataset: 'RGB_image_single' # choose the datasets you want to use.
# Option: "single_RGB_image", "single_fusion_image", "multi_IR_image", ...
mode: 'visible' # 'IR', 'Fusion'
...
```

### network
If you want to use network, you can add it in src/models.

## Datasets
Here I show how we store input dataset for single object detection, you can create a new file under file "datasets".

```
  DATAROOT
  └── datasets
      └── file_name
          ├── thermal
          │   ├── 00000.jpg
          │   ├── 00001.jpg
          │   ├── 00002.jpg
          │   ├── ...
          │   └── ...
          ├── visible
          │   ├── 00000.jpg
          │   ├── 00001.jpg
          │   ├── 00002.jpg
          │   ├── ...
          │   └── ...
          └── label.csv
        

```

```
Single object training framework
└── datasets



  DATAROOT
  └── datasets
      └── file_name
          ├── thermal
          │   ├── 00000.jpg
          │   ├── 00001.jpg
          │   ├── 00002.jpg
          │   ├── ...
          │   └── ...
          ├── visible
          │   ├── 00000.jpg
          │   ├── 00001.jpg
          │   ├── 00002.jpg
          │   ├── ...
          │   └── ...
          └── label.csv
        

```