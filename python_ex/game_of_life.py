import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n=100 #size

# starting board
A = np.random.choice(a=[0,1], size=n*n).reshape(n,n)

movements = [(1,1), (1,0), (1, -1), (0,1), (0, -1), (-1,0), (-1,1), (-1,-1)] # equals to  [(i,j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0)]

#create black and white image
fig = plt.figure()
# use axes instead of im = plt.imshow in order to disable x,y ticks and not opening two windows
ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
im = ax.imshow(A, cmap='Greys', interpolation='nearest',  animated=True)

def next_step(*args):
    global A
    #sum any cell's neighbours
    sum_square = sum(np.roll(np.roll(A, i, 0), j, 1) for (i, j) in movements)

    # Any cell cell with exactly three live neighbours becomes alive
    three_cond = sum_square == 3
    # A dead cell with  two live neighbours becomes alive (and a live cell stays alive)
    two_cond = A & ((sum_square == 2))
    # on other cases, any cell will die (0)

    # it is enough that one of the condition exists
    A = two_cond | three_cond

    ## code to show the board in binary way
    # sys.stdout.write('\033[H')  # move to the top
    # sys.stdout.write('\033[J')  # clear the screen
    # time.sleep(.5)
    # print A

    im.set_data(A)

    return im,


ani = animation.FuncAnimation(fig, next_step, interval=150, blit=True)
plt.show()
