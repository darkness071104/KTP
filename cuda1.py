import numpy as np
import numba
from numba import cuda

# Fungsi CUDA untuk menghitung jumlah kemunculan elemen tertentu dalam array
@cuda.jit
def count_element(arr, target, result):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x

    count = 0
    for i in range(tid, arr.size, stride):
        if arr[i] == target:
            count += 1

    cuda.atomic.add(result, 0, count)  # Penjumlahan atomik untuk menghindari kondisi perlombaan

# Ukuran array dan inisialisasi
array_size = 10
arr = np.random.randint(0, 10, array_size).astype(np.int32)  # Contoh array acak
target_element = 5
result = np.zeros(1, dtype=np.int32)

# Konfigurasi blok dan grid
block_size = 256
grid_size = (array_size + block_size - 1) // block_size

# Memanggil fungsi CUDA
count_element[grid_size, block_size](arr, target_element, result)

# Mencetak setiap prosesnya
for i in range(array_size):
    print(f"Elemen ke-{i}: {arr[i]}")

# Mencetak jumlah kemunculan elemen tertentu
print(f"Jumlah kemunculan elemen {target_element}: {result[0]}")