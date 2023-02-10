import math
import time

nplayers = 20
# size_c = 2

for size_c in range(nplayers+1):
    one = int((math.factorial(nplayers - 1) / (math.factorial(size_c) * math.factorial(nplayers - 1 - size_c))))
    two = math.comb(nplayers - 1, size_c)

    print(size_c, one, two, one == two)


# start1 = time.time()
# for _ in range(100000000):
#     (math.factorial(nplayers - 1) / (math.factorial(size_c) * math.factorial(nplayers - 1 - size_c)))
# end1 = time.time()
#
# start2 = time.time()
# for _ in range(100000000):
#     math.comb(nplayers - 1, size_c)
# end2 = time.time()


print('custom', end1 - start1)
print('math', end2 - start2)
