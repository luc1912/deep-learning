import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
    def __init__(self):
        minx = 0    
        maxx = 10
        miny = 0
        maxy = 10

        self.mu = np.random.random_sample(2) * [maxx-minx, maxy-miny] + [minx, miny]
        eigvalx = (np.random.random_sample()*(maxx - minx)/5)**2
        eigvaly = (np.random.random_sample()*(maxy - miny)/5)**2
        D = np.diag([eigvalx, eigvaly])
        theta = np.random.random_sample()*np.pi*2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.sigma = R.T @ D @ R

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mu, self.sigma, n)
        

def sample_gauss_2d(C, N):
    for i in range(C):
        G = Random2DGaussian()
        X = G.get_sample(N)
        if i == 0:
            x = X
            y = np.ones(N) * i
        else:
            x = np.vstack((x, X))
            y = np.hstack((y, np.ones(N) * i))
    return x, y


def eval_perf_binary(Y, Y_):
    tp = np.sum(np.logical_and(Y == 1, Y_ == 1))
    tn = np.sum(np.logical_and(Y == 0, Y_ == 0))
    fp = np.sum(np.logical_and(Y == 0, Y_ == 1))
    fn = np.sum(np.logical_and(Y == 1, Y_ == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, precision, recall


def eval_AP(ranked_labels):
    """Recovers AP from ranked labels"""
  
    n = len(ranked_labels)
    pos = sum(ranked_labels)
    neg = n - pos
    
    tp = pos
    tn = 0
    fn = 0
    fp = neg
    
    sumprec=0
    for x in ranked_labels:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)    

        if x:
            sumprec += precision

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec/pos


def graph_data(X,Y_, Y, special=[]):
  """Creates a scatter plot (visualize with plt.show)

  Arguments:
      X:       datapoints
      Y_:      groundtruth classification indices
      Y:       predicted class indices
      special: use this to emphasize some points

  Returns:
      None
  """
  palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
  colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
  for i in range(len(palette)):
    colors[Y_==i] = palette[i]

  sizes = np.repeat(20, len(Y_))
  sizes[special] = 40
  
  good = (Y_==Y)
  plt.scatter(X[good,0],X[good,1], c=colors[good], 
              s=sizes[good], marker='o', edgecolors='black')

  bad = (Y_!=Y)
  plt.scatter(X[bad,0],X[bad,1], c=colors[bad], 
              s=sizes[bad], marker='s', edgecolors='black')
  

def graph_surface(function, rect, offset=0.5, width=256, height=256):
  """Creates a surface plot (visualize with plt.show)

  Arguments:
    function: surface to be plotted
    rect:     function domain provided as:
              ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot

  Returns:
    None
  """

  lsw = np.linspace(rect[0][1], rect[1][1], width) 
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

  values=function(grid).reshape((width,height))
  
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  
  plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval)
    
  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

if __name__=="__main__":
    np.random.seed(100)
  
    X,Y_ = sample_gauss_2d(2, 100)
    Y = myDummyDecision(X)>0.5
  
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0)
    graph_data(X, Y_, Y)
    plt.show()