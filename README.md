# DL20Questions

## Overview
DL20Questions is a repository for an implementation of a Natural Language Processing (NLP) system that plays the game 20 Questions. This project aims to demonstrate the capabilities of NLP in understanding and generating human-like questions in the context of the game.

<div id="flow" align="center">
  <img src="https://github.com/shirotman/DL20Questions/blob/main/assets/20Q-gameflow.PNG" width="700"/>
</div>

## Repository Structure
- **local-datasets**: This folder contains the datasets we have created and used to train our models.
- **fine-tuned-models**: This folder contains our trained models.
- **models-code**: This folder contains the Python scripts for training and using our NLP models. These scripts are used to fine-tune and analyze our models.
- **utils**: This folder contains the script used to preprocess the data from the datasets mentioned above.
- **"home" directory**:
  - **game-flow.py**: This is the main script that uses the models and allows us to play the game with the artificial player.
  - **DL20Questions_Report.pdf**: Our final project report.
  - **requirements.txt**: The packages required for running our code.

## Prerequisites
|Library         | Version |
|----------------------|----|
|Python|  3.8  |
|torch|  2.2.2 |
|numpy|  1.24.2 |
|peft|  0.10.0 |
|transformers|  4.39.3 |
|matplotlib|  3.8.0 |
|datasets|  2.18.0 |
|sentence_transformers| 2.6.1 |

## Usage
To run the game, use the script ```game-flow.py``` located in this directory:
```Shell
python game-flow.py
```

**Before** running the game, you need to train the GPT2 (Asker) model **from the ```models-code``` directory**:  
```Shell
cd models-code
python GPT2_ft2prompts.py
```

**Optional (not required for running the game as the repository contains a checkpoint):** For training the FLAN-T5 (Predictor) model, run the script ```flan-t5-LoRA.py``` **from the ```models-code``` directory**:
```Shell
cd models-code
python flan-t5-LoRA.py
```

Eventually, all models will be located in the ```fine-tuned-models``` directory.  

* GPT2_Inference.py:  
Inside the ```models-code``` directory there is a file named ```GPT2_Inference.py``` which is used to test the questions asker (GPT2) and provide perplexity analysis.  In order to run it make sure to run ```GPT2_ft2prompts.py``` first, which generates the fine tuned GPT2 model.
```Shell
cd models-code
python GPT2_Inference.py
```
       
**Game teaser:**
<div id="flow" align="center">
  <img src="https://github.com/shirotman/DL20Questions/blob/main/assets/20Q_game_example.png" width="400"/>
</div>
