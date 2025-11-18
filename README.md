# StreamFlow

To speed up our decoder (based Rectified Flow), we developed a library based on [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) to accelerate the Rectified Flow model, which can achieve a speedup of 300% to 600% and supports unlimited multi-GPU decoding.

We are probably one of the few open-source tools that accelerate RF processing. We wrote approximately 30,000 to 100,000 lines of code to improve this acceleration library.