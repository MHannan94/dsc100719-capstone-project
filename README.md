### dsc100719 Module 7 Capstone Project

# Handwriting recognition in Bengali

#### Author- Mohammed Hannan

## Project description

The <a href= https://en.wikipedia.org/wiki/Bengali_alphabet>Bengali alphabet</a> consists of over 150 letters. A word is made up of characters, and characters are constructed from multiple letters. Specifically, a character contains a grapheme root, and can have two additional accents: vowel and consonant diacritics. The challenge is to create a machine learner that can identify the components in a single grapheme.

## Data
Collected by Bengali.AI, the data consists of over 200,000 images of handwritten graphemes collected from volunteers filling out surveys.

### Files
<b>train.csv</b>
<UL>
    <LI><code>image_id</code>: the foreign key for the image file</LI>
    <LI><code>grapheme_root</code>: the first target class</LI>
    <LI><code>vowel_diacritic</code>: the second target class</LI>
    <LI><code>consonant_diacritic</code>: the third target class</LI>
    <LI><code>grapheme</code>: the complete character (for illustration purposes only)</LI>
</UL>

<b>train.parquet files</b>
Each parquet file contains tens of thousands of 137x236 grayscale images.
<UL>
    <LI><code>image_id</code>: foreign key for the train.csv file</LI>
    <LI> pixel values for image in vector form (1x32,332)</LI>
</UL>

<b>class_map.csv</b>
Maps the class labels to the actual grapheme components

