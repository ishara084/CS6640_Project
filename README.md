# CS6640 - Final Project


## Folder structure
```
|-- root
    |-- Model_results_visualization.ipynb                             ------------> Comparative results summarization and visualization of all models
    |-- utils.py                                                      ------------> Contain common class and methods for data transformation/preparation and evaluation metrics calculation
    |-- Basic_CNN_Model.ipynb                                         ------------> Basic CNN Architecture for testing(Not included in the report)
    |-- Basic_EDA.ipynb                                               ------------> Basic Data summarization
    |-- TransferLearning_yolo.ipynb                                   ------------> Yolo architech for testing(Not included in the report)
    |-- TrasnferLearning_AlexNet.ipynb                                ------------> Pretrained AlexNet Model
    |-- TrasnferLearning_GoogleNet_Inception_V1.ipynb                 ------------> Pretrained GoogleNet(Inception v1) Model
    |-- TrasnferLearning_GoogleNet_Inception_v3.ipynb                 ------------> Pretrained GoogleNet(Inception v3) Model
    |-- TrasnferLearning_ResNet50.ipynb                               ------------> Pretrained ResNet50 Model
    |-- TrasnferLearning_ResNet50_local_pretrained_model.ipynb        ------------> Pretrained ResNet50 Model (For testing only. Not included in the report)
    |-- data                                                          ------------> Contain original data
    |   |-- test.csv                                                  ------------> Contains image name, ground thruth for Test data
    |   |-- train.csv                                                 ------------> Contains image name, ground thruth for Training data
    |   |-- images_train                                              ------------> Training ISW image set
    |   |   |-- 1.png
    |   |   |-- 100.png
    |   |   |-- ......
    |   |-- images_test                                               ------------> Test ISW image set
    |   |   |-- 1.png
    |   |   |-- 100.png
    |   |   |-- ......
    |-- model_outputs_data                                            ------------> Directory for model output data(For saving the best performed model after training, which can be later used if needed.)
    |   |-- best_alexnet_model.pth
    |   |   |-- ......
    |   |-- model_evaluation_logs                                     ------------> Directory for saving model training data for later evaluations
    |   |   |-- training_logs_alexnet.csv
    |   |   |-- .....
    |   |-- model_prediction_logs                                     ------------> Directory for saving model predictions for later analysis
    |   |   |-- alexnet_labels_predictions.csv
    |   |   |-- .....
    |   |-- yolo_output                                               ------------> Output directory for data preparation of YOLO model
    |-- pre_trained_models                                            ------------> Locally saved pre-trained models ( Not included for the experiment. Just for testing)
    |   |-- resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    |   |-- yolov8x-cls.pt
    |-- runs                                                          ------------> Output directory for saving model training of  YOLO model (Automatically created)
```

## Notes
  - Data transformation/processing is done in utils.py to maintain the consistency
  - Each model has a separate notebook with annotations. This is to tweak the model architecture if needed.
  - In model training loop, the best version of the model is saved based on validation loss if I need to train it more later.
  - Training evaluation metrics are logged in ```model_outputs_data/model_evaluation_logs``` directory  for comparisons.
  - YOLO was also included in the code as a test and it performed well(F1 = 0.94), but it was not included in the report because it's often considered as a segmentation model rather than classification
  - Another two models (VGG16, SqueezeNet) was tested and performance was really low compared to other models. Did not include them because;
    - Training time was very long(VGG16 specifically) making it difficult to test the model and do necessary adjustment
    - It's difficult to conclude their performance without further investigations, therefore this was ignored.
    - Their implementation(files) is in the code base, but ignore them.

##  Steps to execute
- NOTE: Trained models and pretrained model files are not pushed because of file size limitation(You can execute the code to generate them, if needed)
- Install necessary libraries before executing the notebook files
  - ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124```
  - ```pip install tensorflow[and-cuda]```
  - ```pip install -U scikit-learn```
  - ```pip install pandas```
  - ```pip install matplotlib```
  - ```pip install ptflops```

Source of the Images & Access Conditions
- Dataset used: https://www.kaggle.com/competitions/internal-waves/data
- Dataset sourced from: https://xwaves.ifremer.fr/#/quicklook
- Use of the images is regulated by: http://en.data.ifremer.fr/All-about-data/Data-access-conditions