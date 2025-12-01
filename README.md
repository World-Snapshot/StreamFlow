# StreamFlow

[[Paper]](https://world-snapshot.github.io/StreamFlow/static/paper/StreamFlow.pdf) [[Project Page]](https://world-snapshot.github.io/StreamFlow/)

To speed up our [Rectified-Flow-based](https://github.com/lqiang67/rectified-flow) project, we developed a library based on [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) to accelerate the RF model, which can achieve a speedup of 300% to 600% of the original speed, and supports unlimited multi-GPU decoding.

![StreamFlow Teaser](./static/videos/StreamFlow_teaser.gif)

# StreamFlow: Theory, Algorithm, and Implementation for High-Efficiency Rectified Flow Generation

We are probably one of the few open-source tools that accelerate RF processing, and we are the first work to systematically optimize and accelerate the Rectified Flow model.


## Env

cd StreamFlow repo,

```code
conda create -n StreamFlow python=3.11 -y
conda activate StreamFlow
pip install -r env/requirements.txt
pip install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```

If you encounter any problems, you can also try the .backup environment.

## Usage

```python
## Default: TensorRT is off.
python test_demo_gen.py
```

## About Work

**Code:** 1. This StreamFlow dependency library is frequently updated, so some of the code may change frequently. 2. Since the World Snapshot organization relies on this acceleration library for many of its projects, and each project has its own adjusted variant, it is very difficult to organize. 

Therefore, the current public version is a relatively early and clean version. If there are any differences between the code and the paper, the code shall prevail. We may carry out major version updates from time to time in the future.

**Compatibility:** We did not conduct any further experiments on other flow models. If there are slight differences in their time steps, they need to be adapted independently.

**Example:** We are currently in the process of cleaning up, but we have some tasks that are of higher priority. If you want to use it in various scenarios as soon as possible (eg, img2img), you can develop it yourself. You can study the example of StreamDiffusion and then modify it to call our library. 

## Q&A

- In rare cases, changing the default number of steps may result in generation errors. These issues will be addressed and resolved in future updates.

## Contact

If you are interested in matters such as the code, how to collaborate, please just contact [@FangSen9000](https://github.com/FangSen9000).

## Acknowledgments

Thank all the authors, colleagues and friends for their work and discussions. Thank [@ZL](https://github.com/ZonglinL) for developing a multi-GPU example.

We have noticed that some optical flow estimation and data processing works have the same abbreviated names as ours, but their essence is different from ours (they operate on optical flow or data flow, while we accelerate the recently emerging Rectified Flow).

Finally, we would like to express our gratitude to [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion). We developed based on their work.

## BibTex

```code
@misc{fang2025streamflowtheoryalgorithmimplementation,
      title={StreamFlow: Theory, Algorithm, and Implementation for High-Efficiency Rectified Flow Generation}, 
      author={Sen Fang and Hongbin Zhong and Yalin Feng and Dimitris N. Metaxas},
      year={2025},
      eprint={2511.22009},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.22009}, 
}
```