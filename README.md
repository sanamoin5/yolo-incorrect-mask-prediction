# Face Mask Detection using YoloV3
 
The system is based on Real-Time Face Mask Detection Method Based on YOLOv3 using Properly Wearing Masked Face Detection Dataset (PWMFD). 

Modifications to the original script([Darknet](https://github.com/pjreddie/darknet)):
Following changes are made before the execution:
- New utility py file to convert the annotation files (.xml) in the required Yolo format (.txt with
normalization).
- Modified Yolo configuration file classes and filter parameter. Filter size is calculated using: number
of anchors * (5+ number of classes). In our case, 3*(5+3) = 24.
- Number of batches, epochs, and subdivisions are modified due to GPU memory constraints.
- For executing the model, Pretrained convolutional layer embeddings are downloaded and the darknet
repository is cloned, along with few modifications in Makefile to make use of GPU, CUDA, and
OPENCV.
- A specific folder structure for data is used(images and labels folder under data with the list of files
under train.txt and val.txt for training and validation respectively) as per Yolo's execution constraints.
- Finally, a script is created to collect the final outputs on the detection set and create the required
visualizations as well for the error analysis.