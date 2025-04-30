# LeggedExplorer

## Installation
**Step Pre:** Only test on Ubuntu22.04

**Step I:** Follow the [Isaac Lab official documentation](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html) to install the Isaac Sim 4.5.0 and Isaac Lab 2.1.0. (isaacsim4.2+isaaclab1.4.1 last commit: 9a246cd9593751c2f98c7bf47f4be0650104d548)

**Step II:** Install [ROS2 Humble](https://docs.ros.org/en/humble/index.html) with the official installation guide.

## Todo
**USD scene:** Matterport3d, Nvidia Omniverse, ...

**Task extensions(Isaaclab):** locomotion, navigation, vln(follow Isaac-VLNCE), ...

**locomotion:** himloco(isaacgym), wmp(isaacgym), H-Infinity(not opensource yet), legged_loco, ...

**local_planner:** NavRL, Viplanner, ...

**VLN/VLA_explorer:** NaVILA, TagMap/VLMaps(pre-built-map&habitat-sim), InstructNav(habitat-sim), ... 

**SLAM+Navigation Baseline:**

## Simulation Test
**locomotion weights copy:** check locomotion/ckpts folder

**usd resources copy and env variable set:** check customed usd folder and
```shell
export USER_PATH_TO_USD=/home/lenovo/Documents/usd
```

**settings:** simulation/sim.yaml
