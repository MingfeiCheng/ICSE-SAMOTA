import os
import subprocess
import multiprocessing
import time
import sys
from road_factory import get_road
import config as cfg
from os import path
from data_handler import get_values
import copy
import glob
# import pyautogui
def run_command(cmd):
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline()

        # Do something else
        return_code = process.poll()
        if return_code is not None:
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                print(output.strip())
            break
        return return_code


def run_command_without_printing(cmd):
    process = subprocess.Popen(cmd,universal_newlines=True)
    return_code = process.poll()


def stop_pylot_container():
    cmd = ['docker', 'stop', 'pylot']
    run_command(cmd)


def start_pylot_container():
    cmd = ['docker', 'start', 'pylot']
    run_command(cmd)


def start_simulator(tree, buildings):
    if tree == 0 and buildings == 0:
        cmd = ['nvidia-docker', 'exec', '-i', '-t', 'pylot',
               '/home/erdos/workspace/pylot/scripts/run_simulator_without_t_b.sh']
    elif tree == 1 and buildings == 0:
        cmd = ['nvidia-docker', 'exec', '-i', '-t', 'pylot',
               '/home/erdos/workspace/pylot/scripts/run_simulator_without_b.sh']
    elif tree == 0 and buildings == 1:
        cmd = ['nvidia-docker', 'exec', '-i', '-t', 'pylot',
               '/home/erdos/workspace/pylot/scripts/run_simulator_without_t.sh']
    else:
        cmd = ['nvidia-docker', 'exec', '-i', '-t', 'pylot', '/home/erdos/workspace/pylot/scripts/run_simulator.sh']
    run_command_without_printing(cmd)


def remove_finished_file():
    cmd = ['docker', 'exec', 'pylot', 'rm', '-rf', '/home/erdos/workspace/results/finished.txt']

    run_command_without_printing(cmd)
    if path.exists("finished.txt"):
        os.remove("finished.txt")


def handle_container(fv):
    stop_pylot_container()
    start_pylot_container()
    remove_finished_file()
    p = multiprocessing.Process(target=start_simulator, name="start_simulator", args=(fv[13], fv[14]))
    p.start()
    time.sleep(35)


def handle_pylot():
    cmd = [cfg.base_directory+'./pylot_runner.sh']
    run_command_without_printing(cmd)


def read_base_file():
    base_file = open(cfg.base_file_name, "r")
    return base_file.read()


def update_file_contents(fv, file_contents):

    file_contents = file_contents + "#####SIMULATOR CONFIG##### \n"

    file_contents = get_road(fv,file_contents)

    file_contents = file_contents + "\n--vehicle_in_front=" + str(fv[3])
    file_contents = file_contents + "\n--vehicle_in_adjcent_lane=" + str(fv[4])
    file_contents = file_contents + "\n--vehicle_in_opposite_lane=" + str(fv[5])
    file_contents = file_contents + "\n--vehicle_in_front_two_wheeled=" + str(fv[6])
    file_contents = file_contents + "\n--vehicle_in_adjacent_two_wheeled=" + str(fv[7])
    file_contents = file_contents + "\n--vehicle_in_opposite_two_wheeled=" + str(fv[8])
    weather = ""
    if (fv[9] == 0):  # noon

        if (fv[10] == 0):  # clear
            weather = "ClearNoon"
        if (fv[10] == 1):  # clear
            weather = "CloudyNoon"
        if (fv[10] == 2):  # clear
            weather = "WetNoon"
        if (fv[10] == 3):  # clear
            weather = "WetCloudyNoon"
        if (fv[10] == 4):  # clear
            weather = "MidRainyNoon"
        if (fv[10] == 5):  # clear
            weather = "HardRainNoon"
        if (fv[10] == 6):  # clear
            weather = "SoftRainNoon"
    if (fv[9] == 1):  # sunset

        if (fv[10] == 0):  # clear
            weather = "ClearSunset"
        if (fv[10] == 1):  # clear
            weather = "CloudySunset"
        if (fv[10] == 2):  # clear
            weather = "WetSunset"
        if (fv[10] == 3):  # clear
            weather = "WetCloudySunset"
        if (fv[10] == 4):  # clear
            weather = "MidRainSunset"
        if (fv[10] == 5):  # clear
            weather = "HardRainSunset"
        if (fv[10] == 6):  # clear
            weather = "SoftRainSunset"
    if (fv[9] == 2):  # sunset

        if (fv[10] == 0):  # clear
            weather = "ClearSunset"
        if (fv[10] == 1):  # clear
            weather = "CloudySunset"
        if (fv[10] == 2):  # clear
            weather = "WetSunset"
        if (fv[10] == 3):  # clear
            weather = "WetCloudySunset"
        if (fv[10] == 4):  # clear
            weather = "MidRainSunset"
        if (fv[10] == 5):  # clear
            weather = "HardRainSunset"
        if (fv[10] == 6):  # clear
            weather = "SoftRainSunset"
        file_contents = file_contents + "\n--night_time=1"

    file_contents = file_contents + "\n--simulator_weather=" + weather
    num_of_pedestrians = 0
    if fv[11] == 0:
        num_of_pedestrians = 0
    if fv[11] == 1:
        num_of_pedestrians = 18

    file_contents = file_contents + "\n--simulator_num_people=" + str(num_of_pedestrians)

    file_contents = file_contents + "\n--target_speed=" + str(fv[12] / 3.6)

    log_file_name = '/home/erdos/workspace/results/' + str(fv)
    file_contents = file_contents + "\n--log_fil_name=" + log_file_name

    return file_contents


def move_config_file():
    cmd = ['docker', 'cp', 'e2e.conf', 'pylot:/home/erdos/workspace/pylot/configs/']
    run_command(cmd)


def handle_config_file(fv):
    file_contents = read_base_file()

    file_contents = update_file_contents(fv, file_contents)

    e2e_File = open(cfg.base_directory+"e2e.conf", "w")
    e2e_File.write(file_contents)
    e2e_File.close()
    move_config_file()


def scenario_finished():
    cmd = [cfg.base_directory+'./pylot_finish_file_copier.sh']
    run_command_without_printing(cmd)
    if path.exists(cfg.base_directory + "finished.txt"):
        return True
    return False


def copy_to_host(fv):
    import os
    if not os.path.exists(cfg.base_directory + "replay_res/"):
        os.makedirs(cfg.base_directory + "replay_res/")
    file_name = 'pylot:/home/erdos/workspace/results/' + str(fv)  # + '.log'
    cmd = ['docker', 'cp', file_name, cfg.base_directory + "replay_res/"]
    run_command(cmd)
    file_name = 'pylot:/home/erdos/workspace/results/' + str(fv)   + '_ex.log'
    cmd = ['docker', 'cp', file_name, cfg.base_directory + "replay_res/"]
    run_command(cmd)

def run_single_scenario(fv_arg):

    fv = copy.deepcopy(fv_arg)
    if fv[12]< 10:
        fv[12] = fv[12]*10

    handle_container(fv)
    handle_config_file(fv)
    handle_pylot()
    counter = 0
    while (True):
        counter = counter + 1
        time.sleep(1)
        if (counter > cfg.time_allowed or scenario_finished()):
            stop_pylot_container()
            break
        else:
            print("Waiting for scenario to finish")
    copy_to_host(fv)
    file_name = 'replay_res/' + str(fv)
    if os.path.exists(file_name):
        DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = get_values(fv)
        return DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max
    else:
        return 1000,1000,1000,1000,1000,1000
def run(fv):

    handle_container(fv)

    handle_config_file(fv)
    handle_pylot()
    counter = 0
    while (True):
        counter = counter + 1
        time.sleep(1)
        if (counter > cfg.time_allowed or scenario_finished()):
            stop_pylot_container()
            break;
        else:
            print("Waiting for scenario to finish")
    copy_to_host(fv)

def main(replay_cfg):
    cfg_str = os.path.basename(replay_cfg)
    line = cfg_str.split("[")[1]
    vector = line.split("]")[0]
    line_parts = vector.split(", ")
    print(line_parts)
    fv = [
        int(line_parts[0]),
        int(line_parts[1]),
        int(line_parts[2]),
        int(line_parts[3]),
        int(line_parts[4]),
        int(line_parts[5]),
        int(line_parts[6]),
        int(line_parts[7]),
        int(line_parts[8]),
        int(line_parts[9]),
        int(line_parts[10]),
        int(line_parts[11]),
        int(line_parts[12]),
        int(line_parts[13]),
        int(line_parts[14]),
        int(line_parts[15])
    ]

    if not os.path.exists('output/replay_res'):
        os.makedirs('output/replay_res')

    file_writer = open(
                    "output/replay_res/" + cfg_str + '_res', 'w')
    
    DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = run_single_scenario(fv)

    file_writer.write(str(fv)+","+str(DfC_min) + "," + str(DfV_max) + "," + str(DfP_max) + "," + str(DfM_max) + "," + str(
        DT_max) + "," + str(traffic_lights_max) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Replay Simulation')
    parser.add_argument('--config', type=str, help='Test config yaml file.', required=True)
    args = parser.parse_args()
    
    main(args.config)
