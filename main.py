import numpy as np
import numba.cuda as cuda

# Fungsi kernel untuk menghitung jumlah kemunculan elemen tertentu dalam array
@cuda.jit
def countOccurrences(array, element, partialResults):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    count = 0

    if tid < array.size:
        if array[tid] == element:
            count = 1

    # Menyimpan hasil perhitungan pada blok yang sesuai
    cuda.atomic.add(partialResults, cuda.blockIdx.x, count)

def main():
    array = np.array([1, 5, 6, 8, 2, 8, 7, 8, 3, 4, 13, 9], dtype=np.int32)
    elementToCount = 8  # Ubah ini menjadi elemen yang ingin Anda hitung kemunculannya
    arraySize = array.size

    # Konfigurasi jumlah blok dan benang
    threadsPerBlock = 256
    blocksPerGrid = (arraySize + threadsPerBlock - 1) // threadsPerBlock

    # Alokasi memori untuk hasil per blok
    partialResults = np.zeros(blocksPerGrid, dtype=np.int32)

    # Mengukur waktu eksekusi pada GPU
    start = cuda.event()
    stop = cuda.event()
    start.record()

    # Panggil kernel
    countOccurrences[blocksPerGrid, threadsPerBlock](array, elementToCount, partialResults)

    # Mengukur waktu eksekusi pada GPU
    stop.record()
    stop.synchronize()
    milliseconds = cuda.event_elapsed_time(start, stop)

    # Salin hasil per blok dari perangkat ke host
    result = np.sum(partialResults)

    # Tampilkan jumlah kemunculan elemen dalam array
    print(f"Jumlah kemunculan elemen {elementToCount} dalam array: {result}")

    # Tampilkan waktu eksekusi pada GPU
    print(f"Waktu eksekusi pada GPU: {milliseconds} ms")

    # Tampilkan hasil per blok
    print("Hasil per blok:")
    for i, count in enumerate(partialResults):
        print(f"Blok {i}: {count}")

if __name__ == "__main__":
    main()