# ASTER
Agent for Story Turn-taking using Event Representations

Code from the paper LJ Martin, P Ammanabrolu, X Wang, W Hancock, S Singh, B Harrison, and MO Riedl. Event Representations for Automated Story Generation with Deep Neural Nets, Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), New Orleans, LA.
http://laramartin.net/pub/AAAI18-EventRepresentations.pdf

## Data:
Can be found here: https://www.dropbox.com/s/a7hhmex8yd6mvcs/EventRepresentationDataRelease.tar.gz?dl=0

Event-to-Event and Event-to-Sentence files can be found in their respective folders. Within each, data files are located in folders that are numbered in the order that they are presented in the paper. Each of these nested folders contains files for test input, expected test output, and generated output (labeled accordingly).

## Running the code:
**Event-to-Sentence** code is run using an [Anaconda](https://www.anaconda.com/download/#linux "Anaconda 2") environment for Python 2.7. The environment is defined in** Event-to-Sentence/environment.yml**. Run `conda env create -f environment.yml` and then `source activate aster` to enter the correct environment.

All configurations can be done in **config.json**. Data is formatted as bi-text (one text file with the input sequences and another with the corresponding output sequences aligned by line number). Run the code using `python nmt.py --config config.json` to train and `python decode.py --config config,json` to decode.

**Event-to-Event** code relies on Tensorflow 1.3, edit hyperparameters in **cmu_translate.py** and run using python 2.7 to train the model and use **cmu_translate_decode.py** to decode.



