import subprocess
from threading import Thread
import time
import sys

class Power_Monitor_Thread(Thread):
    def __init__(self, sleep_time):
        super().__init__()
        if(sleep_time > 0.0):
            self.sleep_time = sleep_time
        else:
            self.sleep_time = 0.0
        self.total_avg_power = 0.0
        self.num_runs = 0
        self.running = True

    def terminate(self):
        self.running = False

    def get_results(self):
        if(self.num_runs == 0):
            return "0 runs"
        else:
            print("Num_Runs {}".format(self.num_runs))
            return (self.num_runs,
                    self.total_avg_power / self.num_runs)

class GPU_Power_Monitor_Thread(Power_Monitor_Thread):
    def __init__(self, sleep_time, device_id):
        super().__init__(sleep_time)
        self.device_id = device_id
        self.total_queries = 0
        self.total_max_power = 0.0
        self.total_min_power = 0.0

    def run(self):
        while(self.running):
            (max_watt, min_watt, avg_watt, queries) = self.get_power()
            self.total_max_power = self.total_max_power + max_watt
            self.total_min_power = self.total_min_power + min_watt
            self.total_avg_power = self.total_avg_power + avg_watt
            self.num_runs = self.num_runs + 1
            self.total_queries = self.total_queries + queries
            time.sleep(self.sleep_time)
    
    def get_power(self):
        command = "nvidia-smi --query --id="+str(self.device_id)+" --display=POWER"
        try:
            if(sys.version_info >= (3, 7)):
                result = subprocess.run(command, check=True, capture_output=True, universal_newlines=True, shell=True)
            else:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        result = str(result).split(":")
        #print(result)
        max_watt = float(result[-3].split(" W")[0])
        min_watt = float(result[-2].split(" W")[0])
        avg_watt = float(result[-1].split(" W")[0])
        queries =  int(result[-4].split("\\n")[0])
        return (max_watt, min_watt, avg_watt, queries)

    def get_results(self):
        if(self.num_runs == 0):
            return "0 runs"
        else:
            print("Num_Runs {}".format(self.num_runs))
            return (self.total_queries,
                    self.total_avg_power / self.num_runs)


class AGX_Power_Monitor_Thread(Power_Monitor_Thread):
    def __init__(self, sleep_time):
        super().__init__(sleep_time)

    def run(self):
        while(self.running):
            avg_watt = self.get_power()
            self.total_avg_power = self.total_avg_power + avg_watt
            self.num_runs = self.num_runs + 1
            time.sleep(self.sleep_time)

    def get_power(self):
        command = "tegrastats | head -n 1"
        try:
            if(sys.version_info >= (3, 7)):
                result = subprocess.run(command, check=True, capture_output=True, universal_newlines=True, shell=True)
            else:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        split_result = result.stdout.split("/")
        SYS5V = int(split_result[-2].split(" ")[-1])
        VDDRQ = int(split_result[-3].split(" ")[-1])
        CV = int(split_result[-4].split(" ")[-1])
        SOC = int(split_result[-5].split(" ")[-1])
        CPU = int(split_result[-6].split(" ")[-1])
        GPU = int(split_result[-7].split(" ")[-1])
        return float(GPU + CPU + SOC + CV + VDDRQ + SYS5V)/1000.0

class Alveo_Power_Monitor_Thread(Power_Monitor_Thread):
    def __init__(self, sleep_time, device_id):
        super().__init__(sleep_time)
        self.device_id = device_id
    
    def run(self):
        while(self.running):
            avg_watt = self.get_power()
            self.total_avg_power = self.total_avg_power + avg_watt
            self.num_runs = self.num_runs + 1
            time.sleep(self.sleep_time)

    def get_power(self):
        command = "xbutil examine -d "+str(self.device_id)+" --r electrical"
        try:
            if(sys.version_info >= (3, 7)):
                result = subprocess.run(command, check=True, capture_output=True, universal_newlines=True, shell=True)
            else:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        split_result = result.stdout.split("Power")
        watt = float(split_result[2].split(" ")[-4])
        return watt


class ARM_Power_Monitor_Thread(Power_Monitor_Thread):
    def __init__(self, sleep_time):
        super().__init__(sleep_time)

    def run(self):
        while(self.running):
            avg_watt = self.get_power()
            self.total_avg_power = self.total_avg_power + avg_watt
            self.num_runs = self.num_runs + 1
            time.sleep(self.sleep_time)

    def get_power(self):
        command = "tegrastats | head -n 1"
        try:
            if(sys.version_info >= (3, 7)):
                result = subprocess.run(command, check=True, capture_output=True, universal_newlines=True, shell=True)
            else:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        split_result = result.stdout.split("/")
        SYS5V = int(split_result[-2].split(" ")[-1])
        VDDRQ = int(split_result[-3].split(" ")[-1])
        CV = int(split_result[-4].split(" ")[-1])
        SOC = int(split_result[-5].split(" ")[-1])
        CPU = int(split_result[-6].split(" ")[-1])
        GPU = int(split_result[-7].split(" ")[-1])
        return float(GPU + CPU + SOC + CV + VDDRQ + SYS5V)/1000.0


class CPU_Power_Monitor_Thread(Power_Monitor_Thread):
    def __init__(self, sleep_time):
        pass
