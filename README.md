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

 
