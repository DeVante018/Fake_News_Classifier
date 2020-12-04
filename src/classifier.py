# this is the main file to run to test if something is fake news or not
import image_classifier
import text_classifier

if __name__ == '__main__':
    text_predictor = text_classifier.train_text()
    #image_predictor = image_classifier.train_images()
    print(" process done...")
