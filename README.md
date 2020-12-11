# Fake_News_Classifier

This project is a classifier that contains code for training and classifying sources of material. This model can classify images, text or both to determine the nature of a source. 

Below you will find a guide for environment setup and packages needed for installation. For all environments you need to be using Python ver3.5 - ver3.8

**SETUP in Anaconda (recommended)**

Install Anaconda

install the Jupiter notebook

Create a new environment and name it (If you're using mac you may not need to do this)

Once the environment is done, change the top header to "not installed" and search for Tensorflow

Click on it and then click apply in the lower right (do the same for the Keras package)

Once these are finished installing, go back to home and open the jupiter notebook with the new environment ( make sure the "Application on" field is set
to the name of new environment)

Search for where the downloaded project is and open it


**SETUP IntelliJ**

**NOTE:** THIS IS NOT COMPATIBLE WITH INTELLIJ COMMUNITY EDITON


In the IntelliJ environment install these packages

------------------------------PACKAGES------------------------------

image classifier: matplotlib, cv2(Open Computer Vision), os,numpy

text classifier: pandas,torch,transformers and sklearn

------------------------------PACKAGES------------------------------

Also, manually install Tensorflow. Use:

pip install --upgrade tensorflow


**HOW TO USE**

To use, simply run the Multimodal_image_text_class.py file

Import all sub-packages that are required of the py file

Be sure to change the file paths of the code on lines 47, 48, 49. The new path should be the location of the file on your hardware.

                                              ^
                                              |
                                              |
                                              |
                                              |
                                              V
                                              
 **Download files here: https://drive.google.com/drive/folders/1hxscceigHP0UxBmRYQBuT-QRz6bxTzwL?usp=sharing
 
**NOTE** windows users should type in r before the file path for the program to find it

*Ex: r'C:\Users\person\project_location\Fake_News_Classifier\...'

Our 3 classification are as follows: 

0 - True

1 - Misleading

2 - Parody/Satire

**Trouble Shooting**

If Your having issues setting up the environments or libraries please reference: https://www.youtube.com/watch?v=O8yye2AHCOk
