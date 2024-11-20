# CS6640 - Final Project


### 1. Folder structure
```
|-- root
    |-- Basic_CNN_Model.ipynb                                         ------------> Basic CNN Architecture for testing(Not included in the report)
    |-- Basic_EDA.ipynb                                               ------------> Basic Data summarization
    |-- TransferLearning_yolo.ipynb                                   ------------> Yolo architech for testing(Not included in the report)
    |-- TrasnferLearning_AlexNet.ipynb                                ------------> Pretrained AlexNet Model
    |-- TrasnferLearning_GoogleNet_Inception_V1.ipynb                 ------------> Pretrained GoogleNet(Inception v1) Model
    |-- TrasnferLearning_GoogleNet_Inception_v3.ipynb                 ------------> Pretrained GoogleNet(Inception v3) Model
    |-- TrasnferLearning_ResNet50.ipynb                               ------------> Pretrained ResNet50 Model
    |-- TrasnferLearning_ResNet50_local_pretrained_model.ipynb        ------------> Pretrained ResNet50 Model (For testing only. Not included in the report)
    |-- TrasnferLearning_SqueezeNet.ipynb                             ------------> Pretrained SqueezeNet Model
    |-- TrasnferLearning_VGG16.ipynb                                  ------------> Pretrained VGG16 Model
    |-- data                                                          ------------> Contain original data
    |   |-- test.csv                                                  ------------> Contains image name, ground thruth for Test data
    |   |-- train.csv                                                 ------------> Contains image name, ground thruth for Training data
    |   |-- images_train                                              ------------> Training ISW image set
    |   |   |-- 1.png
    |   |   |-- 100.png
    |   |   |-- ......
    |   |   |-- 952.png
    |   |-- images_test                                               ------------> Test ISW image set
    |   |   |-- 1.png
    |   |   |-- 100.png
    |   |   |-- ......
    |   |   |-- 231.png
    |-- model_outputs_data                                            ------------> Directory for model output data(For saving the best performed model after training, which can be later used if needed.)
    |   |-- best_alexnet_model.pth
    |   |-- best_googlenet_inception_v3_model.pth
    |   |-- best_googlenet_model.pth
    |   |-- best_resnet50_model.pth
    |   |-- best_squeezenet_model.pth
    |   |-- best_vgg16_model.pth
    |   |-- model_evaluation_logs                                     ------------> Directory for saving model training data for later evaluations
    |   |   |-- training_google_inception_v3_net_logs.csv
    |   |   |-- training_logs_alexnet.csv
    |   |   |-- training_resnet50_logs.csv
    |   |   |-- training_squeezenet_logs.csv
    |   |-- yolo_output                                               ------------> Output directory for data preparation of YOLO model
    |-- pre_trained_models                                            ------------> Locally saved pre-trained models ( Not included for the experiment. Just for testing)
    |   |-- resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    |   |-- yolov8x-cls.pt
    |-- runs                                                          ------------> Output directory for saving model training of  YOLO model (Automatically created)
```

### 2. Steps
- NOTE: Trained models and pretrained model files are not pushed because of file size limitation(You can execute the code to generate them, if needed)
- Install necessary libraries before executing the notebook files
  - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  - pip install tensorflow[and-cuda]
  - pip install -U scikit-learn
  - pip install pandas
  - pip install matplotlib
  - pip install ptflops
