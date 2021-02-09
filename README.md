# Image-Segmentation-Project
Semantic Image segmentation Peach and apple blossoms
The concern focuses on determining the best configuration settings for  semantic segmentation of images of apple, peach and pear flowers under different conditions. The thinking is that before we can extend a classification model to determine bloom intensity we should be able to build a system that can accurately segment images into blooms and non-blooms or classify the blooms under different orchard conditions. 

We propose to deploy the following Keras based segmentation models from the Segmentation Models API : 
•	Unet
•	FPN
•	Linknet
•	PSPNet

1.	The COCO-Stuff  dataset will be used for model pre-training. This dataset provides annotations of classed such as grass, leaves, trees and flowers.

2.	US Department of Agriculture (USDA) has made available the training dataset used to train the image classifying algorithm outlined in Dias et al. This dataset will be used to train the models we intend to experiment with. The dataset comprises four(4) sets of flower images and their accompanying ground truth(labels). The data set is currently arranged as follows:
•	AppleA.zip
•	AppleA_Labels.zip

•	AppleA_Train.zip
•	AppleA_Labels_Train.zip  -
•	
•	AppleB.zip
•	AppleB_Labels.zip

•	Peach.zip
•	Peach_Labels.zip

•	Pear.zip
•	Pear_labels.zip
This USDA dataset will be used to train test and validate our pre-trained models.
