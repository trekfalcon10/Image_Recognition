# Image_Recognition

### The Problem

Can we utilize machine learning to correctly identify images by training on pre-labeled images that were previously labeled by a different machine?  In other words, can machines teach machines to recognize images?  

### The Client

This topic would be of interest to robotics firms to assist in robotic navigation, to law enforcement agencies (to assist in the facial recognition of suspects) or to intelligence agencies (to aid in recognizing targets of interest in aerial reconnaissance photographs).  If successful, it may obviate or reduce the need to have large numbers of human beings pre-label images in every case.  Instead, a client could save money if image recognition training could always be done based on pre-labeled images done by machines that were in turn trained by other machines that were ultimately trained on only one or two human pre-labeling sessions long ago.  
	
### The Data

The dataset was taken from the Open Images dataset available [here](https://storage.googleapis.com/openimages/web/download.html).   The full set consists of 9,178,275 images with 19,995 classes.  There is also a smaller subset of the data consisting of 1,743,042 training images with 601 classes. The data for the image labels and image id’s are available as csv files.  

### The Approach

1.	Data wrangling and cleaning—First I determined which of the machine-learned labels and id’s in the training set correspond to the training labels and id’s for the image subset given and narrowed the training set down to only those labels and id’s.  (The machine learned training set was taken from all 9,178,275 images, while the images used here will be from a more manageable subset.)  I also made certain that I narrowed the set further to only those labels that were identified with certainty by a human observer as confirmation in addition to the machine's estimate/assignment of the label so as not to introduce too much uncertainty into the present model.    
2.	Exploratory Data Analysis—Here, I briefly analyzed relationships among the variables in the dataset using both statistical functions and graphical analyses. This primarily involved an analysis of the frequencies and types of labels present in the dataset, as these were the variables most amenable to analysis.
3.	Machine Learning—I then built a deep learning model using a convolutional neural network to identify the images.  The object was to determine whether and to what extent one deep learning model may accurately recognize images that have been labeled by a different machine and to see how confident both machines were in the identification.
4.	Final Report—Attached, I have provided a detailed report describing the procedure I used and the findings I obtained therefrom, including appropriate data visualizations.  Accompanying this report is a slide deck providing a high level summary. 

### Contents

1.  Image_Recognition.ipynb--The final report and code with output in Jupyter Notebook form
2.  Image_Recog_Presentation.pptx--PowerPoint presentation summarizing the final report
3.  Image_Recognition-Milestone_Report.ipynb--Draft report and code in Jupyter Notebook form, prior to implementation of Machine Learning
4.  Image_Milestone_Presentation.pptx--Draft PowerPoint presentation summarizing the draft report

  #### CSV files from source site
  1.  https://storage.googleapis.com/openimages/2018_04/train/train-annotations-machine-imagelabels.csv
  2.  https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv
  3.  https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels-boxable.csv
  
  #### Image files from source site using Amazon Web Services Download
  
  Use the below command to download the relevant image files to your directory after installing [awscli](https://aws.amazon.com/cli/). 
  
  	aws s3 --no-sign-request sync s3://open-images-dataset/train [target_dir/train]
  
  #### Miscellaneous Additional Asset files
  
  The below supporting files may be downloaded from [here](https://www.dropbox.com/sh/yr0tkw0jgbefia9/AABqb6iTJUjyD7bgBrTKg2u4a?dl=0).
  
  1.  label_train_merge.pkl --Pickled file of cleaned and merged dataset
  2.  extracts_train.pkl  --Pickled file of feature and label extraction for VGG16 model
  3.  extracts_x_train.pkl  --Pickled file of feature and label extraction for first Xception model 
  4.  extracts_xC.pkl  --Pickled file of feature and label extraction for final Xception model 
  5.  history.pkl  --Pickled file of training history for VGG16 model 
  6.  hist_x.pkl  --Pickled file of training history for first Xception model
  7.  hist_XC.pkl --Pickled file of training history for final Xception model
  8.  im_test_XC.joblib  --Joblib file of test images and labels
  9.  new_model_1.h5 --HDF5 file of trained VGG16 model
  10.  new_model_x.h5 --HDF5 file of trained first Xception model
  11. new_model_xC.h5 --HDF5 file of trained final Xception model
  

