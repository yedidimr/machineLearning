'''
This code runs an Expectation Maximization (EM) example on a simple Gaussian mixture model.
The code uses the SkLearn EM implementation built into its Gaussian Mixture Model (GMM) 
implementation and Matplotlib plotting tools.

@author: kvlinden
@version Jan 14, 2013
'''

from matplotlib.patches import Ellipse
from matplotlib.pyplot import scatter, show, figure
from sklearn import mixture
from sklearn.mixture.gmm import GMM
import numpy
import math

# Create a Gaussian mixture model with the given number of components.
components = 1
g = mixture.GMM(n_components=components)

#-----------------------------
# Generate n data points for a single Gaussian with the given mu and sigma.
n = 100
mu = [0.5, 0.5]
sigma = [0.4, 0.6]
data = sigma * numpy.random.randn(n, 2) + mu
print numpy.random.randn(1,2)

#-----------------------------
# Learn (and print) an appropriate Gaussian mixture model.
g.fit(data)
print 'Weights:\n' + str(g.weights_)
print 'Means:\n' + str(g.means_)
print 'Covariance:\n' + str(g.covars_)  # This gives sigma^2.

#-----------------------------
# Use the learned model to predict the values of some sample data.
# samples = numpy.array([[0.2, 0.8], [0.5, .5], [0.8, 0.8], [0.5, 0.75]])
# print g.predict(samples)

#-----------------------------
# Plot the data.
fig = figure()
ax = fig.add_subplot(111)

# Add the raw data points.
ax.scatter(data[:, 0], data[:, 1], color='blue')

# Add the learned mean values.
ax.scatter(g.means_[:, 0], g.means_[:, 1], color='red')

# Add ellipses showing the standard deviation distances from the mean (three levels).
for i in range(components):
    for j in range(1, 4):
        e = Ellipse(g.means_[i],
                    width=math.sqrt(g.covars_[i][0]) * j,
                    height=math.sqrt(g.covars_[i][1]) * j,
                    fill=False, color='red', linestyle='dashed', linewidth=0.5)
        ax.add_artist(e)

# scatter(samples[:, 0], samples[:, 1], color = 'red')

show()
