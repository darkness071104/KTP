import numpy as np
from numba import cuda
import random
import time

# Fungsi kernel CUDA untuk perkalian berulang bilangan integer
@cuda.jit
def integer_multiplication_kernel(base, exponent, result):
    idx = cuda.grid(1)
    if idx < base.shape[0]:
        res = base[idx]
        for i in range(1, exponent):
            res *= base[idx]
            result[idx, i] = res

# Fungsi pemangkatan pada CPU
def integer_power_cpu(base, exponent):
    res = 1
    for i in range(exponent):
        res *= base
    return res

def main():
    # Bilangan integer
    base = random.randint(1, 10)
    max_exponent = random.randint(1, 10)

    # Menyiapkan variabel untuk hasil
    num_results = max_exponent
    result = np.empty((1, num_results), dtype=np.int32)

    # Menentukan jumlah thread per blok
    threads_per_block = 256  # Contoh jumlah thread per blok yang lebih besar

    # Menentukan jumlah blok
    blocks_per_grid = 10  # Contoh jumlah blok yang lebih besar

    # Memindahkan data ke perangkat CUDA
    base_gpu = cuda.to_device(np.array([base], dtype=np.int64))
    result_gpu = cuda.device_array((1, num_results), dtype=np.int32)  # Perbaiki inisialisasi

    # Mengukur waktu eksekusi GPU
    start_gpu = cuda.event()
    end_gpu = cuda.event()

    start_gpu.record()
    
    # Memanggil kernel CUDA
    integer_multiplication_kernel[blocks_per_grid, threads_per_block](base_gpu, max_exponent, result_gpu)

    end_gpu.record()
    end_gpu.synchronize()

    # Mengukur waktu eksekusi CPU
    start_cpu = time.time()

    # Proses pemangkatan pada CPU
    result_cpu = integer_power_cpu(base, max_exponent)

    end_cpu = time.time()
    print(f"Start CPU time: {start_cpu}")
    print(f"End CPU time: {end_cpu}")

    # Mencetak hasil perkalian
    for i in range(max_exponent):
        print(f"{base} ^ {i + 1} = {result_gpu[0, i]}")  # Mengambil hasil dari GPU

    # Mencetak hasil akhir
    final_result = result_gpu[0, max_exponent - 1]  # Mengambil hasil dari GPU
    print(f"Hasil akhir dari {base} ^ {max_exponent} (GPU): {final_result}")

    final_result_cpu = result_cpu
    print(f"Hasil akhir dari {base} ^ {max_exponent} (CPU): {final_result_cpu}")

    # Menghitung waktu eksekusi
    gpu_time = cuda.event_elapsed_time(start_gpu, end_gpu)
    print(f"Waktu eksekusi GPU: {gpu_time} ms")

    cpu_time = (end_cpu - start_cpu) * 1000  # Dalam milidetik
    print(f"Waktu eksekusi CPU: {cpu_time} ms")

if __name__ == "__main__":
    main()