# Fair bandits

## Install
All commands are to be run from the unzipped directory unless specified otherwise.

### Create virtual environment
``` python
conda create --name fairbandit pip
conda activate fairbandit
```

### Requirements
``` bash
pip install -r requirements.txt
```

For the core algorithm
* numpy

For plotting and running experiments
* matplotlib
* joblib

For running tests
* pytest

### Install fairbandits
``` bash
pip install -e .
```

Run tests
``` bash
pytest
```


## Experiments

### Reproducing Figure 1: Benchmark of LCB, UCB and Greedy on a MAB problem 

Move into the `experiments` directory and run the benchmark:

`python mab.py` 

Move into the `plotting` directory and plot the data:

`python mab.py` (Computation time: 1 second )

Move into the `figure` directory to see the reproduced figure:

![Figure 1](./figures/mab.png)


### Reproducing Figure 2: Benchmark of LCB, UCB and Greedy on a contextual MAB problem 

Move into the `experiments` directory and run the benchmark:

`python contextual_mab.py` 

Move into the `plotting` directory and plot the data:

`python contextual_mab.py` (Computation time: 1 second )

Move into the `figure` directory to see the reproduced figure:

![Figure 2](./figures/contextual_mab.png)


