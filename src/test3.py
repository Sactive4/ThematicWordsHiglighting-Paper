import numpy as np

a = [[1, 23, 4], [54, 65, 95]]
b = [12, 232, 42, 542, 652, 952]
c = [13, 233, 43, 543, 653, 953]
d = [[[1, 2], [3, 4], [9, 10]]] #, [[5, 6], [7, 8], [11, 12]]]

# print(list(zip(a,b,c)))
# print(np.shape(a))
print(np.concatenate(d))
