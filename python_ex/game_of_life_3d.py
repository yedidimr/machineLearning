import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

n=100 #size

# starting board
A = np.random.choice(a=[0,1], size=n*n*n).reshape(n,n,n)

movements = equals to  [(i,j,k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i != 0 or j != 0 or k!=0)]

#create black and white image
fig = plt.figure()
# use axes instead of im = plt.imshow in order to disable x,y ticks and not opening two windows
ax = fig.add_axes([0, 0, 0, 1, 1, 1], xticks=[], yticks=[], zticks=[], frameon=False, projection='3d')


# todo: load 3d matrix
# im = ax.imshow(A, cmap='Greys', interpolation='nearest',  animated=True)

def next_step(*args):
    global A
    #sum any cell's neighbours
    sum_square = sum(np.roll(np.roll(np.roll(A, i, 0), j, 1), k, 2) for (i, j, k) in movements)

    # todo: fill conditions

    im.set_data(A)

    return im,


ani = animation.FuncAnimation(fig, next_step, interval=150, blit=True)
plt.show()
