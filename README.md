# Chaos to Knowledge: Noise-Entropy Augmentation in Robotic Applications of Reinforcement Learning

This research examines the integration of entropy regularization and noise incorporation to advance reinforcement learning for high-dimensional tasks in robotic environments. By treating noise and entropy as interconnected elements of an informational framework, and combining them we develop a novel model that exhibits greater resilience to environmental disturbances and superior navigation of action possibilities. Evaluations in MuJoCo settings contrast baseline TRPO with shannon entropy term modulated by a coheficient hyperpameter, showing that entropy-infused approaches expand action diversity, boost disturbance tolerance, and match algorithmic strengths to task demands. Outcomes confirm that linking entropy with noise yields a stable equilibrium between discovery and reliability, with enhanced models thriving in intricate scenarios. These insights advance the creation of flexible reinforcement learning systems for practical robotic use.

## Usage
To use this project: run `make install` for installation, `make experiments` to execute the paper's experiments, and `make report` to generate and visualize results.

## Citation
If you find this work useful in your research, please consider citing the following paper:

```
@article{green2025chaos,
  title = {Chaos to Knowledge: Noise-Entropy Augmentation in Robotic Applications of Reinforcement Learning},
  author = {Simon Green and Abdulrahman Altahhan},
  institution = {School of Computing, University of Leeds, UK},
  year = {2025},
  note = {Preprint}
}
```
