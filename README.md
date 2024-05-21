---
title: Food Classifier
emoji: 🎂
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.4
app_file: app.py
python_version: 3.10.14
pinned: true
---

# Food image classification

## Intro

This repo was created for personal educational purposes. I used subset of [Food101](https://www.kaggle.com/datasets/dansbecker/food-101) dataset keeping only 3 types of food (targets) to classify. In the project were used different parameters and models to visualize enough experiments in TensorBoard. Also I used gradio and huggingface space as my hosting for the best model.

## Deployment 
[Click here](https://huggingface.co/spaces/blago7daren/Food101_classification) to see the app. <i>(P.s. Or run app.py to host it on your local machine)</i>

## Technologies Used

- PyTorch
- TensorBoard
- Scikit-learn
- Gradio

## Installation and Setup (using "conda 24.1.2")
```bash
git clone git@github.com:nikitazuevblago/Food101_classification.git
cd Food101_classification
conda env create -f environment.yml
activate Food101_classification
tensorboard --logdir=experiments
```
At the end of executing the commands in the terminal, you will be prompted to go to the local server to track the experiments obtained earlier after running **"modeling.ipynb"**

(P.s. Training time - 14 min in Kaggle on P100 GPU)

## Result 
Tracked set of metrics with a possibility to filter them using regular expressions.
(P.s. Regular expressions highlighted with red)

#### Example №1
![result_example_1.png](result_example_1.png)

#### Example №2
![result_example_2.png](result_example_2.png)

## File structure
- **modeling.ipynb** - all experiment tracking process and modeling
- **app.py** - GUI using gradio
- **class_to_idx.pkl** - dictionary with predicted labels
- **environment.yml** - all dependencies
- **models** - folder with models
- **experiments** - folder with tracked experiments


## Possible problems:
1. **Error while downloading model on MAC**: URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>. <br /> **Solution**: https://stackoverflow.com/questions/68275857/urllib-error-urlerror-urlopen-error-ssl-certificate-verify-failed-certifica
2. **RegEx on Windows**: doesn't show filters using one slash. <br /> **Solution**: use double slash on Windows "\\\\", single slash on Mac "/"
3. **Activate conda environment**: usage: conda-script.py [-h] [--no-plugins] [-V] COMMAND .... <br /> **Solution**: 
```bash
conda activate Food101_classification
```
instead of 
```bash
activate Food101_classification
```