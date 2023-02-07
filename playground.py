import random
import time
import ray

ray.init()


@ray.remote
class TestClass:

    def __init__(self):
        pass

    def work(self, i):
        time.sleep(random.random())
        return i


tests = [TestClass.remote() for _ in range(10)]
print('tests')
print(tests)

sum_in_completion_order = 0
refs = [test.work.remote(i) for i, test in enumerate(tests)]
print('refs')
print(refs)

unfinished = refs
while unfinished:
    # Returns the first ObjectRef that is ready.
    finished, unfinished = ray.wait(unfinished, num_returns=1)
    if finished[0] in refs:
        print('finished', finished, refs.index(finished[0]))
    else:
        print('finished', finished)
    result = ray.get(finished[0])
    # process result
    sum_in_completion_order = sum_in_completion_order + result