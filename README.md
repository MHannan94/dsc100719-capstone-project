### dsc100719 Module 7 Capstone Project

# Handwriting recognition in Bengali

#### Author- Mohammed Hannan

## Project description

The <a href= https://en.wikipedia.org/wiki/Bengali_alphabet>Bengali alphabet</a> consists of 49 letters and 18 possible diacritics (or accents). Each character in a word is constructed from up to three components. Specifically, a character contains a grapheme root, and can have two additional accents: vowel and consonant diacritics. There are 168 classes for the grapheme root, 11 classes for the vowel diacritic and 7 classes for the consonant diacritic.  The challenge is to create a machine learner that can identify the components in a single grapheme.

## Data
Collected by Bengali.AI, the data consists of over 200,000 images of handwritten graphemes collected from volunteers filling out surveys. You can download the data from <a href= https://www.kaggle.com/c/bengaliai-cv19/data>here</a>.

### Files
<b>train.csv</b>
<UL>
    <LI><code>image_id</code>: the foreign key for the image file</LI>
    <LI><code>grapheme_root</code>: the first target class</LI>
    <LI><code>vowel_diacritic</code>: the second target class</LI>
    <LI><code>consonant_diacritic</code>: the third target class</LI>
    <LI><code>grapheme</code>: the complete character (for illustration purposes only)</LI>
</UL>

<b>train.parquet files</b><br>
Each parquet file contains tens of thousands of 137x236 grayscale images.
<UL>
    <LI><code>image_id</code>: foreign key for the train.csv file</LI>
    <LI> pixel values for image in vector form (1x32,332)</LI>
</UL>

<b>class_map.csv</b><br>
Maps the class labels to the actual grapheme components.

There are approximately 13,000 unique combinations of the components. The data contains different variations of 1000 graphemes. A good model should be able to classify the other combinations.

### Preparation

To prepare the data, I concatenated the parquet files and merged with the train.csv file on the image ID. I normalized the data so that the pixel values are between 0 and 1 (division by 255). I extracted the pixel values to create the training data and created dummy variables for the three target variables. Instead of using all 32,000 pixels to train my network, I resized the images to 64x64 (4,000 pixels) to reduce memory usage and training time. In order to pass the images through a convolutional neural network, the input data must be of the shape (batch_size, height, width, number_of_channels). The batch size is the number of images and the number of channels is 1 for grayscale images.

I split the data into a training and testing set, with a validation set within the training data, to detect overfitting. I started with a simple network with a couple of convolution layers and a normalization layer. I also used three fully connected layers for the target variables.

## Results

My initial model had difficulty classifying the root character. I expected this to be the biggest hurdle because of the large number of class labels in this target. The accuracy of the vowel and consonant diacritic was a lot higher, likely due to the smaller number of class labels. The network was overfitting to the training data, so I planned to include more dropout layers, generalising the model.

## Improving the model

I included more convolution layers to increase the model's ability to detect features in the images. I looked at the performance of the model during each epoch of training, looking for indications of overfitting. The model performed well on the validation set, giving a root accuracy of 92.15%, vowel accuracy of 98.00% and a consonant accuracy of 97.83%.

## Next steps

My best model took approximately 3 hours to run in the cloud, because of the complexity of the network. To improve the model, I can add more layers, increasing the complexity making it possible to detect smaller differences between graphemes. However, it is important to consider the cost of these additions, the main cost being time.

### Order of notebooks
<OL>
    <LI><code>data_loading_model_1.ipynb</code></LI>
    <LI><code>model_2_3.ipynb</code></LI>
    <LI><code>model_4_5_6_7.ipynb</code></LI>
    <LI><code>final_model_and_results.ipynb</code></LI>
</OL>
