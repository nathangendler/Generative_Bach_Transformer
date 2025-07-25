# Generative Bach Music Transformer Model


## An AI model trained to generative music in the style of Bach


 A quick overview of what the different scripts do:
 - Scrape.py: Converts MusicXML style files to tokens prepped to be fed into model
 - Model.py: Transformer based model that learns to generate new tokens in the style of bach
  - Converter.py: Converts generated tokens back into a MusicXML file for Musescore and other music composition applications

## An example of the generated Music:

![Generated Sheet music!](/sheetMusic/generatedSheetMusic.png)
<br>
<sub><i>*Generated Bach-style sheet music*</i></sub>



<style>
img[alt="Generated Sheet music!"] { 
  width: 600px; 
}
</style>

## How to run it yourself
 While currently this model is currently trained to generated Bach style music, it can learn the compositional style of any other composer:
 - First, replace the current files in the data_file with your MusicXML files of a certain composer and then run the scrape.py script(you can skip this step if you want Bach style music)
 - Then run the model.py file which will train the model and generate a 200 token musicl compositon
 - Finally, run  the converter.py file to convert the generated tokens back into MusicXML


## Future changes I want to make
 - The current model is trained on a very small dataset (only about 3 pieces) so getting access to a larger variety of bach pieces would help improve the model accuracy

