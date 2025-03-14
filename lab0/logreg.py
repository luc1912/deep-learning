import numpy as np
import matplotlib.pyplot as plt
import data

param_niter = 1000
param_delta = 0.1

def logreg_train(X, Y_):

    C = max(Y_) + 1
    N, D = X.shape
    W = np.random.randn(D, C)
    b = np.zeros(C)

    for i in range(param_niter):

        scores = X @ W + b # N x C
        expscores = np.exp(scores) # N x C
        
        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1, keepdims=True) # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp # N x C
        logprobs = np.log(probs) # N x C

        # gubitak
        loss  = -np.mean(logprobs[np.arange(N), Y_])
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        Yoh = data.class_to_onehot(Y_)
        dL_ds = probs - Yoh # N x C

        # gradijenti parametara
        grad_W = X.T @ dL_ds / N # D x C
        grad_b = np.sum(dL_ds, axis=0) / N # C x 1

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b

def logreg_classify(X, W, b):
    scores = X @ W + b # N x C
    expscores = np.exp(scores) # N x C
    sumexp = np.sum(expscores, axis=1, keepdims=True) # N x 1
    probs = expscores / sumexp # N x C

    return probs

def logreg_decfun(W, b):
    def classify(X):
        return np.argmax(logreg_classify(X, W, b), axis=1)
    return classify

if __name__ == "__main__":
    np.random.seed(100)

    C = 3
    N = 100
    X, Y_ = data.sample_gauss_2d(C, N)

    W, b = logreg_train(X, Y_)

    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    acc = np.mean(Y == Y_)
    print("Train accuracy: ", acc)

    decfun = logreg_decfun(W, b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])

    plt.show()