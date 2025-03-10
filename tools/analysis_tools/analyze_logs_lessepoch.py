import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




def main():
  file_path = '/ssd_data/xxy/projects/RSP/Object Detection/work_dirs/time_calculator/DOTA/rvsa_deep/20250108_155356.log.json'
  with open(file_path,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
  print(json_data)
  

if __name__ == '__main__':
    main()
