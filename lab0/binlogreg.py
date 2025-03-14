import numpy as np
import matplotlib.pyplot as plt
import data 

param_niter = 1000
param_delta = 0.1

def binlogreg_train(X,Y_):
    '''
        Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

        Povratne vrijednosti
        w, b: parametri logističke regresije
    '''

    w = np.random.randn(X.shape[1])
    b = 0

    for i in range(param_niter):

        scores = np.dot(X, w) + b     # N x 1
        probs = 1 / (1 + np.exp(-scores))     # N x 1
        loss  = np.mean(-np.log(probs))     # scalar
        
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_     # N x 1

        # gradijenti parametara
        grad_w = dL_dscores.T @ X / X.shape[0]    # 1 x D
        grad_b = np.sum(dL_dscores) / X.shape[0]     # 1 x 1

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    '''
        Argumenti
        X:  podatci, np.array NxD
        w, b: parametri logističke regresije
    
        Povratne vrijednosti
        probs: vjerojatnosti razreda, np.array Nx1
    '''
    scores = np.dot(X, w) + b     # N x 1
    probs = 1 / (1 + np.exp(-scores))     # N x 1
    return probs


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify

# binlogreg_decfun = lambda w, b: lambda X: binlogreg_classify(X, w, b)


if __name__=="__main__":

    np.random.seed(100)

    X, Y_ = data.sample_gauss_2d(2, 100)

    w, b = binlogreg_train(X, Y_)

    probs = binlogreg_classify(X, w, b)
    Y = np.array([1 if p > 0.5 else 0 for p in probs])

    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)


    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()