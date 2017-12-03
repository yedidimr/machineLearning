import random
from threading import Thread

def partition(lst, start, end):
    print start, end
    if start>=end:
        return
    # choose random pivot
    org_pivot_index = end  # random.randint(0, len(lst)-1)
    pivot_index = org_pivot_index
    pivot = lst[pivot_index]

    # order left side
    leftmost = start
    while (leftmost < pivot_index and pivot_index > start):
        if lst[leftmost] > pivot:
            lst[pivot_index] = lst[leftmost]
            lst[leftmost] = lst[pivot_index - 1]
            lst[pivot_index - 1] = pivot
            pivot_index -= 1
        else:
            leftmost += 1

    # # order right side
    # rightmost = org_pivot_index
    # while (rightmost < len(lst)-1 and pivot_index < len(lst)-2):
    #     if lst[rightmost] <= pivot:
    #         lst[pivot_index] = lst[rightmost]
    #         print pivot_index
    #         lst[rightmost] = lst[pivot_index + 1]
    #         lst[pivot_index + 1] = pivot
    #         pivot_index += 1
    #     else:
    #         rightmost += 1
    print "pivot index now is", pivot_index

    t1 = Thread(target=partition, args=(lst, start, pivot_index-1))
    t2 = Thread(target=partition, args=(lst, pivot_index+1, end))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
def quicksort(lst):
    partition(lst, 0, len(lst)-1)
    # lst[1]=11111111


p=[3,7,8,5,2,1,9,5,4]
print p
print len(p)
print "YASY"
quicksort(p)
print p
