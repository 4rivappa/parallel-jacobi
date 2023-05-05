from random import randrange
import sys
import threading
import time
import numpy as np

lock = threading.Lock()

def gen_matrix(size):
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            rand_num = randrange(1,size+1)
            row.append(rand_num)
        # ensure diagonal dominance here:
        row[i] = sum(row) + 1
        matrix.append(row)
    solution = []
    for i in range(size):
        rand_num = randrange(1,size+1)
        solution.append(rand_num)
    return matrix, solution

def jacobi_parallel(A, b, x0, tol=1e-6, max_iter=1000, num_threads=4):
    n = len(b)
    x = x0.copy()
    # print(x)
    # print(x0)
    iteration = 0

    def jacobi_thread(start_idx, end_idx, thread_id):
        for i in range(start_idx, end_idx):
            sum = 0.0
            for j in range(n):
                if j != i:
                    sum += A[i,j] * x[j]
            # lock.acquire()
            x[i] = (b[i] - sum) / A[i,i]
            # lock.release()

    while iteration < max_iter:
        old_x = x.copy()
        threads = []
        chunk_size = n // num_threads
        for i in range(num_threads):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            if i == num_threads - 1:
                end_idx = n
            t = threading.Thread(target=jacobi_thread, args=(start_idx, end_idx, i))
            # print(start_idx, end_idx, i)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        iteration += 1
        # if np.linalg.norm(x - old_x) < tol:
        if np.linalg.norm(np.dot(A, x) - b) < tol:
            break
    # print(iteration)
    if iteration >= max_iter:
        print("Iteration count exceeded.")
    return x, iteration

def get_matrix_from_inputfile(file_name):
    file = open(file_name, 'r')
    matrix = []
    solution = []
    for iter, line in enumerate(file.readlines()):
        if line != "":
            nums = line.strip().split(" ")
            matrix.append(nums[:-1])
            solution.append(nums[-1])
    return matrix, solution


def perform_jacobi_algo(size, threads_count):
    A, b = gen_matrix(size)

    A = np.asarray(A, dtype = np.float64, order ='C')
    b = np.asarray(b, dtype = np.float64, order ='C')
    # print(b)
    # Define the initial guess for the solution
    x0 = np.zeros_like(b)

    # Solve the linear system using the Jacobi algorithm with parallelization
    # if len(sys.argv) == 2:
    #     threads_count = int(sys.argv[1])
    # else:
    #     threads_count = 10
    threads_count = threads_count
    
    start = time.time()
    x, num_iterations = jacobi_parallel(A, b, x0, tol=1e-6, max_iter=10000, num_threads=threads_count)
    end = time.time()

    file = open("results.txt", "a", encoding="utf-8")
    file.write("pthreads " + str(size) + " " + str(threads_count) + " " + str((end-start)*10000))
    file.write("\n")

    # Print the solution and the number of iterations performed
    # print('Solution: ', x)
    # print('Number of iterations: ', num_iterations)
    # print('Time taken is: ' + str(end-start))


if __name__ == '__main__':
    for s in range(100, 1050, 50):
        for t in range(1, 16):
            perform_jacobi_algo(s, t)
            print(s, t)