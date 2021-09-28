import h5py
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
#matplotlib.rcParams.update({'font.size': 15})
mpl.rcParams.update({'figure.dpi': 200})
mpl.rcParams.update({'figure.figsize': (6,4)})
import pandas as pd
import seaborn as sns
from copy import deepcopy

import sys
import os
import subprocess
import glob
from collections import defaultdict
import scipy as sp
from scipy import special
from itertools import product

from scipy.io import savemat

temp = 0.01
pred_files = glob.glob(f"/home/ian/Documents/Data/MD_LV_paper_data/*_pred/*")
val_files = glob.glob(f"/home/ian/Documents/Data/MD_LV_paper_data/*_pred/*")