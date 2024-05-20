# Food image classification

## Intro

This repo was created for personal educational purposes without a goal of implementation. I used subset of [Food101](https://www.kaggle.com/datasets/dansbecker/food-101) dataset keeping only 3 types of food (targets) to classify. In the project were used different parameters and models to visualize enough experiments in TensorBoard.

## Technologies Used

- PyTorch
- TensorBoard
- Scikit-learn

## Installation and Setup
```bash
git clone git@github.com:nikitazuevblago/Food101_classification.git
cd Food101_classification
pip install -r requirements.txt
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

## Possible problems:
1. **Error while downloading model on MAC**: URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>. <br /> **Solution**: https://stackoverflow.com/questions/68275857/urllib-error-urlerror-urlopen-error-ssl-certificate-verify-failed-certifica
2. **RegEx on Windows**: doesn't show filters using one slash. <br /> **Solution**: use double slash on Windows "\\\\", single slash on Mac "/"


# For deployment on Hugging Face
---
title: Food Classifier
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.5.0
app_file: app.py
python_version: 3.12.3
pinned: true
---
