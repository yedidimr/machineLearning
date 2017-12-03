from numpy.linalg import matrix_rank
from matplotlib.pyplot import show, plot, figure
import numpy as np

print("\nHow to get the diagonal of a dot product?\n")
X = np.arange(4).reshape(2,2)
Y = np.arange(4).reshape(2,2)
print np.diag(np.dot(X,Y))

print("\nConsider the vector how to build a new vector with 3 consecutive zeros interleaved between each value?\n")
v = [1,2,3,4,5]
print "orig vector:",v
new = np.zeros(len(v) + (len(v) -1)*3)
new[::4] = v
print new

print("\nCompute averages using a sliding window over an array?\n")
N = 3 #window size
arr=np.array([1.,1.,2.,2.,3.,3.,4.,4.])
print"using array ", arr , "with window size", N
res = arr
for i in range(N-1):
    arr = np.roll(arr,-1)
    res += arr
print res
res = res / N
print "result is", res


print("\nConsider 2 sets of points P0,P1 describing lines and a point p, how to compute distance from p to each line?\n")

print("\nCompute a matrix rank?\n")
matrix_rank(np.ones((4,)))
print ("Ans: using matrix_rank")

print("\nHow to find the most frequent value in an array?\n")
arr=[1,1,4,2,3,4,4,6, 20]
print "array is:", arr
print "most frequent:", np.argmax(np.bincount(arr))

#matplotlib



fig = figure(1)
ax1 = fig.add_subplot(211)

t = np.arange(1, 10, 0.01)
y = np.sin(t)+np.sin(100*t)
ax1.set_title('x=sin(t)+sin(100*t)')
ax1.plot(t,y)

ax2 = fig.add_subplot(212)
## fourier transform
f = np.fft.fft(y)
## sample frequencies
freq = np.fft.fftfreq(len(y), d=t[1]-t[0])
ax2.set_title("Fourier's Transform")
ax2.plot(freq, abs(f)**2) # will show a peak at a frequency of 1 as it should.
show()