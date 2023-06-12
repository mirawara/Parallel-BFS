<h1 align="center" id="title">Parallel BFS</h1>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-C-76B900.svg?style=flat&logo=nvidia&logoColor=white" alt="CUDA C">
    <img src="https://img.shields.io/badge/NVIDIA-GPU-76B900.svg?style=flat&logo=nvidia&logoColor=white" alt="NVIDIA">

</p>


<p id="description">This project is a parallel implementation of the Breadth-First Search (BFS) algorithm for graph traversal using CUDA programming on NVIDIA GPUs. The goal is to improve the performance of BFS on large-scale graphs by exploiting the parallelism of GPUs. The project includes a detailed analysis of the algorithm, implementation, kernel profile, and performance evaluation. The implementation is optimized using various techniques such as memory coalescing, thread work distribution, and shared memory usage. The kernel profile analysis provides insights into the performance bottlenecks and helps in optimizing the implementation further. The performance evaluation is done on various graphs of different sizes and densities, and the results show significant speedup compared to the sequential BFS algorithm on CPU. See the <a href="https://github.com/mirawara/Parallel-BFS/Wiki/BFS-Wiki">Wiki</a> for further details.</p>


<h2>🛠️ Tools for Performance evaluation:</h2>

* <a href="https://developer.nvidia.com/nsight-systems">Nvidia Nsight Systems</a>
* <a href="https://developer.nvidia.com/nsight-compute">Nvidia Nsight Compute</a>

<h2>🖥️ Usage: </h2>
<p>Manual:</p>


```
transcriber.py [-h] -f FILE [-nr NOISE] -o OUT [-iv IV] [-l LANG]

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Path to audio file
  -nr NOISE, --noise-reduction NOISE
                        Noise reduction: there are two levels: level 1 - Basic noise reduction (recommended) level 2 - Massive noise reduction
  -o OUT, --output OUT  Path to output file
  -iv IV, --increase-volume IV
                        Increase volume: you have to provide a float from 0 to 3 in the form int.dec
  -l LANG, --language LANG
                        Language (Default: en-EN)
 ```
 
 <p>Example: </p>
 
 ```
 python3 transcriber.py -f audio_example/Subconscious_Learning.mp3  -o result.txt -nr 1
 ```

<h2>💖Like my work?</h2>

Contact me if you have any corrections or additional features to offer me.
