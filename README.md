# openmp_and_cuda_based_image_processing

## OpenMP
I run OpenMP code on my local machine (MacOS Sequoia 15.3) so I used command line which is compatible to my computer in order to execute the code.

### Execution command
  1. Anaglyph
  ```bash
  /opt/homebrew/opt/llvm/bin/clang++ anaglyph.cpp -fopenmp -isysroot $(xcrun --show-sdk-path) `pkg-config opencv4 --cflags` -c
  /opt/homebrew/opt/llvm/bin/clang++ anaglyph.o -fopenmp -isysroot $(xcrun --show-sdk-path) -stdlib=libc++ `pkg-config opencv4 --libs` -L/opt/homebrew/opt/llvm/lib -lomp -o anaglyph
  ./anaglyph test.jpg true
  ```
  2. Gaussian filter
  ```bash
  /opt/homebrew/opt/llvm/bin/clang++ gaussian_filter.cpp -fopenmp -isysroot $(xcrun --show-sdk-path) `pkg-config opencv4 --cflags` -c
  /opt/homebrew/opt/llvm/bin/clang++ gaussian_filter.o -fopenmp -isysroot $(xcrun --show-sdk-path) -stdlib=libc++ `pkg-config opencv4 --libs` -L/opt/homebrew/opt/llvm/lib -lomp -o   gaussian_filter
 ./gaussian_filter test.jpg true 5 3
  ```
  3. Denoising
  ```bash
  /opt/homebrew/opt/llvm/bin/clang++ denoising.cpp -fopenmp -isysroot $(xcrun --show-sdk-path) `pkg-config opencv4 --cflags` -c
  /opt/homebrew/opt/llvm/bin/clang++ denoising.o -fopenmp -isysroot $(xcrun --show-sdk-path) -stdlib=libc++ `pkg-config opencv4 --libs` -L/opt/homebrew/opt/llvm/lib -lomp -o denoising
  ./denoising noise.png 5 3
  ```

## CUDA
I run CUDA code on VM.

### Execution command
  1. Anaglyph
  ```bash
  /usr/local/cuda-12.1/bin/nvcc -O3 anaglyph.cu `pkg-c onfig opencv4 --cflags --libs` -o anaglyph
  ./anaglyph test.jpg 1
  ```
  2. Gaussian filter
  ```bash
  /usr/local/cuda-12.1/bin/nvcc -O3 gaussian_filter.cu `pkg-c onfig opencv4 --cflags --libs` -o gaussian_filter
 ./gaussian_filter test.jpg 1 5 3
  ```
  3. Denoising
  ```bash
  /usr/local/cuda-12.1/bin/nvcc -O3 denoise.cu `pkg-c onfig opencv4 --cflags --libs` -o denoise
  ./denoise noise.png 5 3
  ```
  4. Gaussian filter with share memory
  ```bash
  /usr/local/cuda-12.1/bin/nvcc -O3 gaussian_share_memory.cu `pkg-c onfig opencv4 --cflags --libs` -o gaussian_share_memory
  ./gaussian_share_memory test.jpg 1 5 3
  ``` 

