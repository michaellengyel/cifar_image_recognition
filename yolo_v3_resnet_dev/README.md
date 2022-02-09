#Yolo V3 Implementation

##Data
Topics: VOC vs. COCO, Calculating anchor boxes.  

##Dataloader
Topics: Creating targets from labels (y), Augmenting data (x) with albumentations. Difference between training  
validation and test datasets.

##Loss
Losses: Object loss, no object loss, class loss, box coordinate loss.  
The format of the loss.   
Optionally try using GIoU: https://giou.stanford.edu/  
Optionally try using CIoU:  

##Model
https://paperswithcode.com/method/darknet-53   
https://stackoverflow.com/questions/65131134/fusing-the-resnet-50-and-yolo-algorithms-for-enhanced-deep-learning-object-detec   
https://ieeexplore.ieee.org/document/9332457

##Unit Tests
https://realpython.com/python-testing/   

##Optimisation
Profile the code to find the bottlenecks. Optimise the code so that the majority of the compute would be the actual  
training. Consider using parallel python libs for cpu operations, e.g. calculating the metrics.

##Training
Hyperparamaters, optimization, support code e.g. saving model and weights, train-time logging with tensorboard.  
Rendering bounding boxes.

Document environment specific decisions: Using multiple GPUs, float16, etc.  

##Evaluation
Intersection Over Union (Jaccard Index), TP, TN, FP, FN, Precision, Recall, Precision Recall Curve, mean Average  
precision (mAP or AP), Confusion Matrix.

Analyse: https://github.com/ultralytics/yolov3/issues/898 
Analyse: https://github.com/ultralytics/yolov5/issues/918

##Benchmarks
A more extensive analysis of the fully trained network to evaluate it's performance.  

#Miscellaneous Topics