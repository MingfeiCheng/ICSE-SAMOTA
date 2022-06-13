# SAMOTA
This project is a fork of [2022-ICSE-Efficient Online Testing for DNN-Enabled Systems using
Surrogate-Assisted and Many-Objective Optimization](https://orbilu.uni.lu/bitstream/10993/50091/1/SAMOTA-authorversion.pdf) (SAMOTA).

I test this project on Ubuntu 20.04 running on Intel i5-11600K CPU with RTX 3060 Ti (8 GB) and 32 GB Memory.

## Environment Config with Docker
I use Pylot docker image with the [latest version](https://github.com/erdos-project/pylot) (Tensorflow 2.5.1) rather than v0.3.2 (Tensorflow 1.5.x) provided in the paper.

### Config Pylot
#### Step1: Download Pylot
```
docker pull erdosproject/pylot
nvidia-docker run -itd --name pylot -p 20022:22 erdosproject/pylot /bin/bash
```

#### Step2: Update Pytorch
Update Pytorch to 1.10 in the docker container.
```
nvidia-docker exec -i -t pylot /bin/bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
Note: I update the Pytorch (0.4.x in the Pylot docker image) to 1.10 with cuda 11.x, because 0.4.x has a compatible problem with RTX 30 series GPU. This issue also exits in Tensorflow 1.5.x, so I use Pylot with the latest version. I do not make sure other series GPUs, such as RTX 20 or GTX 10, have the same issue.

#### Step3: Setup SSH of Pylot Container
```
ssh-keygen
nvidia-docker cp ~/.ssh/id_rsa.pub pylot:/home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo chown erdos /home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo service ssh start
```

#### Step4: (Optional) Check Pylot
Terminal 1:
```
nvidia-docker exec -i -t pylot /home/erdos/workspace/pylot/scripts/run_simulator.sh
```
Terminal 2:
```
nvidia-docker exec -i -t pylot /bin/bash
cd ~/workspace/pylot/
python3 pylot.py --flagfile=configs/detection.conf
```

### Config SAMOTA
#### Step1: Create Conda Env
```
conda create -n samota python=3.7
conda activate samota
pip install -r requirements.txt
```
#### Step2: Download Carla Packages
*download* the simulators from the following links and *extract* them to a folder name `Carla_Versions` (due to the figshare upload limit, the compressed file is divided into three)
* [Link_1](https://doi.org/10.6084/m9.figshare.16443321)
* [Link_2](https://doi.org/10.6084/m9.figshare.16443228)
* [Link_3](https://doi.org/10.6084/m9.figshare.16442883)

#### Step3: Replace Files in the Pylot Container
```
docker cp Carla_Versions pylot:/home/erdos/workspace/
ssh -p 20022 -X erdos@localhost
cd /home/erdos/workspace
mkdir results
logout
docker cp implementation/pylot pylot:/home/erdos/workspace/pylot/
docker cp implementation/scripts pylot:/home/erdos/workspace/pylot/
ssh -p 20022 -X erdos@localhost
cd /home/erdos/workspace/pylot/scripts
chmod +x run_simulator.sh
chmod +x run_simulator_without_b.sh
chmod +x run_simulator_without_t.sh
chmod +x run_simulator_without_t_b.sh
```
