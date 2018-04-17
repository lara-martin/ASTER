# ASTER
Agent for Story Turn-taking using Event Representations

Code from the paper LJ Martin, P Ammanabrolu, X Wang, W Hancock, S Singh, B Harrison, and MO Riedl. Event Representations for Automated Story Generation with Deep Neural Nets, Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), New Orleans, LA.
http://laramartin.net/pub/AAAI18-EventRepresentations.pdf

**Disclaimer** This code is not upkept.

## Data:
Can be found here: https://www.dropbox.com/s/a7hhmex8yd6mvcs/EventRepresentationDataRelease.tar.gz?dl=0

Event-to-Event and Event-to-Sentence files can be found in their respective folders. Within each, data files are located in folders that are numbered in the order that they are presented in the paper. Each of these nested folders contains files for test input, expected test output, and generated output (labeled accordingly).

## Running the code:
**Pruning and Splitting** code
To start a Stanford CoreNLP server, run: `sh runNLPserver.sh`. Run `python corenlp.py` to parse your data and then you can run `python dataCleaning.py` to prune and split.

**Event Creation** code takes separate NER and parse files, and extracts multiple events.

*Note:* The parsing code we have provided has combined the parses and NER into a single file. You will have to change these following files to match this format.
Once you have the parses, you can run `python generalize_events.py` for generalized events, or `python generalize_events_bigrams.py` for event bigrams with continuing named entities, or `python generalize_events_bigrams.py` to generalize the entire sentence.

You can also do topic modeling using LDA (in the folder `Event_Creation/Topic_Modeling`). After you adjust the input file, run `python train_lda.py` to create a model. Once the model is made, run `python lda_classify.py` to find the top words in each topic or `python finidGenre_args.py` to create data files that are separated by these new genres.

**Event-to-Event** code relies on Tensorflow 1.3, edit hyperparameters in **cmu_translate.py** and run using python 2.7 to train the model and use **cmu_translate_decode.py** to decode.


**Event-to-Sentence** code is run using an [Anaconda](https://www.anaconda.com/download/#linux "Anaconda 2") environment for Python 2.7. The environment is defined in **Event-to-Sentence/environment.yml**. Run `conda env create -f environment.yml` and then `source activate aster` to enter the correct environment.

All configurations can be done in **config.json**. Data is formatted as bi-text (one text file with the input sequences and another with the corresponding output sequences aligned by line number). Run the code using `python nmt.py --config config.json` to train and `python decode.py --config config.json` to decode.
