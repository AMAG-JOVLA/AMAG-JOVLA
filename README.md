# Env Set Up

| Environment | Requirement               |
| ----------- | ------------------------- |
| OS          | Must support Egl renderer |
| Python      | 3.8                       |

```sh
conda create -n amag_jovla python=3.8 -y
cd service_dataset_generation
pip install -r requirements.txt
```

# Service Dataset Generation

Download `Partnet-Mobility` from https://sapien.ucsd.edu/.

Extract the zip file to the `dataset`directory.

Start automatically building the dataset.

```sh
./run.sh
```

# Model Finetuning

The key code for model fine-tuning is located in the `model` directory.

```sh
conda create -n vla_model python=3.10 -y
cd model
pip install -r requirements.txt
```

Download checkpoint from [SPHINX](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-v2-1k).

Modify key parameters in `train.sh`. Then start fine-tuning.

```
./train.sh
```

# Experiment

Download Aruco  from [Online ArUco markers generator](https://chev.me/arucogen/).

Print out the Aruco code with a side length of 5 centimeters.

Place it on the end effector of the robotic arm.

Perform hand eye calibration

```sh
python3 calibrate_and_test.py
```

Test grasp

```
python3 grasp.py
```

