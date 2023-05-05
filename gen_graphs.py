
from matplotlib import pyplot as plt
import numpy as np


def load_threads_data():
    file = open("thread_results.txt", "r")
    threads_data = []
    threads_total_data = []
    for line in file.readlines():
        if line == "":
            continue
        content = line.strip().split(" ")
        if content[2] == "4":
            threads_data.append([int(content[1]), float(content[3])])
        threads_total_data.append([int(content[1]), int(content[2]), float(content[3])])
    file.close()
    return threads_data, threads_total_data

def load_cuda_data():
    file = open("cuda_results.txt", "r")
    cuda_data = []
    for line in file.readlines():
        if line == "":
            continue
        content = line.strip().split(" ")
        cuda_data.append([int(content[1]), float(content[2])])
    file.close()
    return cuda_data

def load_serial_data():
    file = open("serial_results.txt", "r")
    serial_data = []
    for line in file.readlines():
        if line == "":
            continue
        content = line.strip().split(" ")
        serial_data.append([int(content[1]), float(content[2])])
    file.close()
    return serial_data

if __name__ == "__main__":
    thread, total_thread = load_threads_data()
    serial = load_serial_data()
    cuda = load_cuda_data()
    
    thread = np.array(thread)
    plt.plot(thread[:, 0], thread[:, 1])
    plt.xlabel('matrix size')
    plt.ylabel('time taken')
    plt.title('size vs time(ms) taken graph for optimal 4 threads')
    plt.show()

    plt.clf()
    serial = np.array(serial)
    plt.plot(serial[:, 0], serial[:, 1])
    plt.xlabel('matrix size')
    plt.ylabel('time taken')
    plt.title('size vs time(ms) taken graph for serial execution')
    plt.show()

    plt.clf()
    cuda = np.array(cuda)
    plt.plot(cuda[:, 0], cuda[:, 1])
    plt.xlabel('matrix size')
    plt.ylabel('time taken')
    plt.title('size vs time(ms) taken graph for cuda execution')
    plt.show()

    plt.clf()
    plt.plot(serial[:, 0], serial[:, 1], label="serial")
    plt.plot(cuda[:, 0], cuda[:, 1], label="cuda")
    plt.xlabel('matrix size')
    plt.ylabel('time taken')
    plt.title('cuda vs serial comparision')
    plt.legend()
    plt.show()

    plt.clf()
    curr_size = 0
    curr_x = []
    curr_y = []
    for row in total_thread:
        # print(row)
        if row[0] == curr_size:
            curr_x.append(row[1])
            curr_y.append(row[2])
        else:
            if curr_size != 0:
                plt.plot(curr_x, curr_y, label=str(curr_size))
            curr_size = row[0]
            curr_x = []
            curr_y = []
            curr_x.append(row[1])
            curr_y.append(row[2])
    plt.xlabel('num of threads')
    plt.ylabel('time taken(ms)')
    plt.title("vis of diff threads vs diff size, optimal thread count is 4")
    plt.legend()
    plt.show()