import urllib2
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2

s1 = urllib2.urlopen('https://www.gutenberg.org/files/1934/1934-0.txt').read()
s2 = file("/home/student-5/PycharmProjects/machineLearning/imageProc/svd.py", 'r').read()


def entropy(data_str):
    data = {c:1.0*data_str.count(c)/len(data_str) for c in set(data_str)}.values()
    return -sum(data * np.log2(data))

def mutual_information(hgram):
    # code is taken from https://matthew-brett.github.io/teaching/mutual_information.html
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


all_chars = set(s1+s2)
s1_dist = {c:1.0*s1.count(c)/len(s1) for c in all_chars}
s2_dist = {c:1.0*s2.count(c)/len(s2) for c in all_chars}

print "code entropy", entropy(s2)
print "poem entropy", entropy(s1)

hist_2d, x_edges, y_edges = np.histogram2d(s1_dist.values(), s2_dist.values(), bins=20)
print "mutual information", mutual_information(hist_2d)
plt.imshow(hist_2d.T, origin='lower')
# plt.hist(s1_dist.values())
# plt.hist(s2_dist.values())
# plt.xlabel('T1 signal bin')
# plt.ylabel('T2 signal bin')
plt.show()
