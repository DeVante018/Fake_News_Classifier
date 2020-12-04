# Fake_News_Classifier

This project is a classifier that contains code for training and classifying sources of material. This model can classify images, text or both to determine the nature of a source. 

Below you will find a guide for environment setup and packages needed for installation. For all environments you need to be using Python ver2.6 or later

SETUP IntelliJ

**NOTE** THIS IS NOT COMPATIBLE WITH INTELLIJ COMMUNITY EDITON

In the IntelliJ enviornment install these packages

------------------------------PACKAGES------------------------------

image classifier: matplotlib, cv2(Open Computer Vision), os,numpy

text classifier: pandas,torch,transformers and sklearn

------------------------------PACKAGES------------------------------

In the code provided the specific packages from these libraries are already installed. 

HOW TO USE

Functions have been provided for model use

•	classify_images(image_location) - images classification 

•	classify_text(tab_sep_sentence) - Text classification (must be tab separated and ended with a newline character)

•	classify_source(tab_sep_sentence, image_location) - image and text classification for a single imae and text

image_location: The file path of the image you want classified
 
tab_sep_sentence: The sentence you want classified type with tabs spacing the words

There are 3 classifications that can be outputed

0 - false

1 - misleading

2 - satire
