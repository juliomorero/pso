# Particle Swarm Optimization (PSO)

The idea of this project is to explore the optimization technique known as Particle Swarm Optimization, initially presented in [Particle swarm optimization - 1995](https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf) and later improved in [A Modified Particle Swarm Optimizer - 1998](https://www.researchgate.net/profile/Yuhui-Shi/publication/3755900_A_Modified_Particle_Swarm_Optimizer/links/54d9a9700cf24647581f009e/A-Modified-Particle-Swarm-Optimizer.pdf?origin=publication_detail).

## Installation

1. After installing [python](https://www.python.org/downloads/release/python-3910/), create and activate a virtual environment
   ```
   > python -m venv <name_of_venv>
   > source <name_of_venv>/bin/activate
   ```
2. Inside the virtual environment install `pip-tools`
   ```
   > python -m pip install pip-tools
   ```
3. Compile production dependencies
   ```
   > pip-compile --allow-unsafe
   ```
4. Install dependencies
   ```
   > pip-sync
   ```

## Usage

### Code used in Guided lecture

1. Check `guided.ipynb`

### Old code

1. Activate virtual environment
   ```
   > source <name_of_venv>/bin/activate
   ```
2. Run pso.py script to optimize [Schaffer's F6 function](https://arxiv.org/pdf/1207.4318.pdf#page=13)
   ```
   > python pso.py
   ```
