#!/usr/bin/pypy

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from pymmh3 import hash

# parameters #################################################################

train = 'train.txt'  # path to training file
test = 'test.txt'  # path to testing file

logbatch = 100000
dotest = True

D = 2 ** 24    # number of weights use for learning

signed = False    # Use signed hash? Set to False for to reduce number of hash calls

interaction = True

lambda1 = 0.
lambda2 = 0.

if interaction:
    alpha = .004  # learning rate for sgd optimization
else:
    alpha = .05   # learning rate for sgd optimization
adapt = 1.        # Use adagrad, sets it as power of adaptive factor. >1 will amplify adaptive measure and vice versa
fudge = .5        # Fudge factor


header = ['Label','i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11','i12','i13','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','c26']

# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-17), 10e-17)        # The bounds
    return -log(p) if y == 1. else -log(1. - p)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(csv_row, D):
    fullind = []
    for key, value in csv_row.items():
        s = key + '=' + value
        fullind.append(hash(s) % D) # weakest hash ever ?? Not anymore :P

    if interaction == True:
        indlist2 = []
        for i in range(len(fullind)):
            for j in range(i+1,len(fullind)):
                indlist2.append(fullind[i] ^ fullind[j]) # Creating interactions using XOR
        fullind = fullind + indlist2

    x = {}
    x[0] = 1  # 0 is the index of the bias term
    for index in fullind:
        if(not x.has_key(index)):
            x[index] = 0
        if signed:
            x[index] += (1 if (hash(str(index))%2)==1 else -1) # Disable for speed
        else:
            x[index] += 1
    
    return x  # x contains indices of features that have a value as number of occurences


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    wTx = 0.
    for i, xi in x.items():
        wTx += w[i] * xi  # w[i] * x[i]
    return 1. / (1. + exp(-max(min(wTx, 50.), -50.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, g, x, p, y):
    for i, xi in x.items():
        # alpha / (sqrt(g) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        delta = (p - y) * xi + lambda1 + lambda2 * w[i]
        if adapt > 0:
            g[i] += delta ** 2
        w[i] -= delta * alpha / (sqrt(g[i]) ** adapt)  # Minimising log loss
    return w, g


# training and testing #######################################################

# initialize our model
w = [0.] * D  # weights
g = [fudge] * D  # sum of historical gradients

# start training a logistic regression model using on pass sgd
loss = 0.
lossb = 0.
for t, row in enumerate(DictReader(open(train), header, delimiter='\t')):
    y = 1. if row['Label'] == '1' else 0.

    del row['Label']  # can't let the model peek the answer

    # main training procedure
    # step 1, get the hashed features
    x = get_x(row, D)
    # step 2, get prediction
    p = get_p(x, w)

    # for progress validation, useless for learning our model
    lossx = logloss(p, y)
    loss += lossx
    lossb += lossx
    if t % logbatch == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), t, loss/t, lossb/logbatch))
        lossb = 0.

    # step 3, update model with answer
    w, g = update_w(w, g, x, p, y)

if not dotest:
    exit()

# testing (build kaggle's submission file)
with open('submission.csv', 'w') as submission:
    submission.write('Id,Predicted\n')
    for t, row in enumerate(DictReader(open(test), header[1:], delimiter='\t')):
        x = get_x(row, D)
        p = get_p(x, w)
        submission.write('%d,%f\n' % (60000000+int(t), p))
