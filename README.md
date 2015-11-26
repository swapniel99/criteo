# criteo
A solution to Criteo Labs Display Advertising Challenge on Kaggle
Link : <https://www.kaggle.com/c/criteo-display-ad-challenge>

This code is derived from the one provided here:
<https://www.kaggle.com/c/criteo-display-ad-challenge/forums/t/10322/beat-the-benchmark-with-less-then-200mb-of-memory>

The above code gives log-loss of about 0.46881 on private leaderboard ranking around 266-297.

--------------------------------------------------

Our code implements Logistic Regression using SGD, Adagrad, Hashing trick and Quadratic interactions

Salient improvements over original code:
* Uses adagrad which accumulates squared gradients instead of number of occurrences of a feature.
* Added a fudge factor for adagrad instead of constant 1. This is initialized only in beginning instead of being added at each step.
* The hashed feature vector x is nolonger a bit map, but saves count of occurences of a feature in an example.
* Used murmur3 hash. For purpose of speeding up with pypy, a python implementation(pymmh3) was used.
* Option to use signed hash instead of unsigned hash. It cancels out collisons by 50%.
* Added option to enable quadratic interaction for pairs of features.

The code as is gives a log-loss of 0.45308 on private leaderboard. Sufficient to place us on 22nd rank.

Lastly, it is recommended to use pypy to run this code.

Disabling interactions improves the speed by 8x taking about 20 mins on my laptop.
With interactions it takes about 4 hours.

pymmh3.py is imported from here:
<https://github.com/wc-duck/pymmh3>

Instructions:

1. Get data from:  <https://www.kaggle.com/c/criteo-display-ad-challenge/data>
2. Rename training file to train.tsv and test file to test.tsv
3. Run criteo.py using pypy

ToDo:
* Add regularization

