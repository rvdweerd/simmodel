import multiprocessing as mp
import numpy as np

# def square(x):
#     return np.square(x)
# x = np.arange(64)
# print(mp.cpu_count())
# pool = mp.Pool(8)
# squared = pool.map(square, [x[8*i:8*i+8] for i in range(8)])
# print(squared)

def square(i, x, queue):
    print("In process {}".format(i,))
    queue.put(np.square(x))

processes = [] #ref to each proccess
queue = mp.Queue() # mp queue, shared across processes
x = np.arange(64)
for i in range(8): # start 8 processes
    start_index = 8*i
    proc = mp.Process(target=square,args=(i,x[start_index:start_index+8], queue))
    proc.start()
    processes.append(proc)
for proc in processes: # wait for all processes to fininsh
    proc.join()
for proc in processes:  # terminate processes
    proc.terminate()
results = []
while not queue.empty():
    results.append(queue.get())
print(results)