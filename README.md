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

DNN:  FC128-FC128-FC10-SoftMax

CNN:  
```
30 epoch in 21000s ('Test score:', 0.20866020459792831)
```

CNN2: CV4(5*5)-CV8(3*3)-MP2*2-CV16(3*3)-MP2*2-FC-128-FC10-SoftMax
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

CNN3: CV4(5*5)-CV8(3*3)-MP2*2-CV16(3*3)-CV32(2*2)-MP2*2-FC128-FC10-Softmax
```
10 epoch in 2400s 
loss: 0.0266 - acc.: 0.9912 - val. loss: 0.0523 - val. acc.: 0.9848
('Test score:', 0.049439467147584119)
```

