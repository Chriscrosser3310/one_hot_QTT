#!/usr/bin/python
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2017 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.0"
# ____________________________________________________________________
# view_density.py
# Author: Dominique Delande 
# Release date: April, 2, 2017
# License: GPL2 or later 
# Tested with Python v2 and Python v3
# Reads 2d data in a ASCII file and make a color 2d plot with matplotlib
# In the ASCII file, the data must be on consecutive lines, following the rows
# of the 2d data, without hole
# The first two lines should contain "# n1 delta1" and "# n2 delta2" where
# n1 (n2) is the number of rows (columns)
# delta1 and delta2 are optional arguments given the step in the corresponding dimension
# Each line may contain several data, the last column is plotted,
# unless the there is a second argument, which is the column to be plotted
# ____________________________________________________________________

def view_density(file_name,column=-1,block=True):
  f = open(file_name,'r')
  #f = open('eigenstate.dat','r')
  line=(f.readline().lstrip('#')).split()
  n1=int(line[0])
  if len(line)>1:
    delta1=float(line[-1])
  else:
    delta1=1.0
  line=(f.readline().lstrip('#')).split()
  n2=int(line[0])
  if len(line)>1:
    delta2=float(line[-1])
  else:
    delta2=1.0
  #print sys.argv,len(sys.argv)
  arr=np.loadtxt(file_name,comments='#').reshape(n1,n2,-1)
  #print arr
  Z=arr[:,:,column]
  print('Maximum value = ',Z.max())
  print('Minimum value = ',Z.min())
  plt.figure()
  plt.imshow(Z,origin='lower',interpolation='nearest')
  plt.show(block)
  return
  
if __name__ == '__main__':
  if (len(sys.argv)!=2) and (len(sys.argv)!=3):
    print('Usage (1 or 2 parameters):\n view_density.py file_name [column]')
    sys.exit()
  file_name=sys.argv[1] 
  if len(sys.argv)==3:
    column=sys.argv[2]
  else:
    column=-1
  view_density(file_name,column)

 