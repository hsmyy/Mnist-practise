# Mnist-practise

## Abstract

This project will use Deep Learning technique to solve mnist digit hand writing problem.

For ease of writing code, I use keras, it's easy and intuitive.

For the DNN version, I borrowed it from keras example. After test, its performance is not good.

Then I tried the CNN version from this article(a little difference):

Multi-column Deep Neural Networks for Image Classification

The performance is exciting.

The next step I want to compare the CNN and LSTM on this dataset and learn more basic idea about it.

## Test

DNN:

CNN:
```
30 epoch in 21000s ('Test score:', 0.20866020459792831)
```

CNN2:
```
10 epoch in 1150s ('Test score:', 0.070545128054809794)
```

CNN2 with relu activation:
```
10 epoch in 1000s ('Test score:', 2.3021199735210596)
```

CNN2 with dropout 0.5 before flatten
```
10 epoch in 960s ('Test score:', 2.3021494342803526)
```
