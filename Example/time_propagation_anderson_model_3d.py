#!/usr/bin/python
from __future__ import print_function
import math
import random
import sys
import os
import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt

__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2017 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.0"
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
# ____________________________________________________________________
#
# time_propagation_anderson_model_3d.py
# Authors: Dominique Delande
# Release date: April, 2, 2017
# License: GPL2 or later 
# Tested with Python v2 and Python v3
# -----------------------------------------------------------------------------------------
# This script models localization in the 3d Anderson model with box disorder, 
# i.e. uncorrelated on-site energies w_n uniformly distributed in [-W/2,W/2].  
# The script propagates in time an initially localized wave packet with a Gaussian
# density with width sigma_0 (usually between 1.0 and 3.0).
# An additional modulation of the wavefunction is added so that the average k vector in each direction 
# is pi/2, so that the energy is on average zero, at the band center. 
# The time propagation is done by expansion on the eigenbasis obtained by numerical diagonalization of the Hamiltonian
# The average squared displacement <r^2(t)> is printed in "squared_displacement.dat"
# and fitted by a power law with background
# The average density probability in configuration space (i.e. |\psi_i|^2) is printed in density.dat
# -----------------------------------------------------------------------------------------
if len(sys.argv) != 6:
  print('Usage (5 parameters):\n time_propagation_anderson_model_3d.py L W nr t_max nsteps')
  sys.exit()
L = int(sys.argv[1])
W = float(sys.argv[2])
nr = int(sys.argv[3])
t_max = float(sys.argv[4])
nsteps = int(sys.argv[5])
t_step = t_max/nsteps
# Size of the initial wavepacket
sigma_0 = 1.0
coef=-0.5/(sigma_0**2)

# Generate a disordered sequence of on-site energies in the array "disorder"
def generate_disorder(L,W):
  disorder=W*((np.random.uniform(size=L*L*L)).reshape((L,L,L))-0.5)
  return disorder

# Generate the Hamiltonian matrix for one realization of the random disorder
# The Hamitonian is in the LxL array "H" 
def generate_hamiltonian(L,W):
  H=np.zeros((L*L*L,L*L*L))
  disorder=generate_disorder(L,W)
  for i in range(L):
    ip1=(i+1)%L
    for j in range(L):
      jp1=(j+1)%L
      for k in range(L):
        kp1=(k+1)%L
        H[i*L*L+j*L+k,i*L*L+j*L+k]=disorder[i,j,k]
        H[ip1*L*L+j  *L+k  ,i  *L*L+j  *L+k  ]=1.0
        H[i  *L*L+j  *L+k  ,ip1*L*L+j  *L+k  ]=1.0
        H[i  *L*L+jp1*L+k  ,i  *L*L+j  *L+k  ]=1.0
        H[i  *L*L+j  *L+k  ,i  *L*L+jp1*L+k  ]=1.0
        H[i  *L*L+j  *L+kp1,i  *L*L+j  *L+k  ]=1.0
        H[i  *L*L+j  *L+k  ,i  *L*L+j  *L+kp1]=1.0
  return H

def generate_initial_state(L):
  psi=np.zeros(L*L*L)
#  phase1=2.0*math.pi*random.random()
#  phase2=2.0*math.pi*random.random()
#  phase3=2.0*math.pi*random.random()
  phase1=0.0
  phase2=0.0
  phase3=0.0
# add a combination of exp(+/-ik) along each direction
# to adjust average energy around 0   
  for i in range(L):
    for j in range(L):
      for k in range(L):
        psi[i*L*L+j*L+k]=math.exp(coef*((i-L/2)**2+(j-L/2)**2+(k-L/2)**2))*math.cos(phase1+0.5*math.pi*(i-L/2))*math.cos(phase2+0.5*math.pi*(j-L/2))*math.cos(phase3+0.5*math.pi*(k-L/2))
# Normalize
  psi*=1.0/math.sqrt(np.sum(psi**2))  
  return psi
 

def compute_r2(psi):
  L=int(psi.size**(1.0/3.0)+0.5)
  r2=0.0
  for i in range(L):
    for j in range(L):
      for k in range(L):
        r2+=((i-L/2)**2+(j-L/2)**2+(k-L/2)**2)*(psi[i*L*L+j*L+k]**2)
  return r2

def power_law(t, background, scaling, exponent):
  return background+scaling*t**exponent

# ________________________________________________
# extract_1d_and_2d_densities.py
# Author: Dominique Delande 
# Release date: April, 2, 2017
# License: GPL2 or later 
# Tested with Python v2 and Python v3
# Reads 3d data in a ASCII file and creates cumulative 1d and 2d densities in various files
# In the ASCII file, the data must be on consecutive lines, following the rows
# of the 3d data, without hole
# The first three lines should contain "# n1 delta1", "# n2 delta2" and "# n3 delta3" 
# where n1, n2, n3 are the dimensions of the array
# delta1, delta2 and delta3 are optional arguments given the step in the corresponding dimension
# Each line may contain several data, the last column is plotted,
# unless the there is a second argument, which is the column to be plotted
# ____________________________________________________________________
def extract_1d_and_2d_densities(density_file_name,column=-1):
  f = open(density_file_name,'r')
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
  line=(f.readline().lstrip('#')).split()
  n3=int(line[0])
  if len(line)>1:
    delta3=float(line[-1])
  else:
    delta3=1.0
  
  #print n1,delta1,n2,delta2,n3,delta3  
  
  arr=np.loadtxt(density_file_name,comments='#').reshape(n1,n2,n3,-1)
  
  Z=arr[:,:,:,column]
  print('Maximum value = ',Z.max())
  print('Minimum value = ',Z.min())
  name, ext = os.path.splitext(density_file_name)
  density_x_file_name=name+'_x'+ext
  g = open(density_x_file_name,'w')
  print('#',n1,delta1,file=g)
  for i in range(n1):
    print(delta1*(i-n1/2),delta2*delta3*np.sum(Z[i,:,:]),file=g)
  g.close()  
  density_y_file_name=name+'_y'+ext
  g = open(density_y_file_name,'w')
  print('#',n2,delta2,file=g)
  for i in range(n2):
    print(delta2*(i-n2/2),delta1*delta3*np.sum(Z[:,i,:]),file=g)
  g.close()  
  density_z_file_name=name+'_z'+ext
  g = open(density_z_file_name,'w')
  print('#',n3,delta3,file=g)
  for i in range(n3):
    print(delta3*(i-n3/2),delta1*delta2*np.sum(Z[:,:,i]),file=g)
  g.close()  
  density_xy_file_name=name+'_xy'+ext
  g = open(density_xy_file_name,'w')
  print('#',n1,delta1,file=g)
  print('#',n2,delta2,file=g)
  for i in range(n1):
    for j in range(n2):
      print(delta1*(i-n1/2),delta2*(j-n2/2),delta3*np.sum(Z[i,j,:]),file=g)
  g.close()  
  density_xz_file_name=name+'_xz'+ext
  g = open(density_xz_file_name,'w')
  print('#',n1,delta1,file=g)
  print('#',n3,delta3,file=g)
  for i in range(n1):
    for j in range(n3):
      print(delta1*(i-n1/2),delta3*(j-n3/2),delta2*np.sum(Z[i,:,j]),file=g)
  g.close()  
  density_yz_file_name=name+'_yz'+ext
  g = open(density_yz_file_name,'w')
  print('#',n2,delta2,file=g)
  print('#',n3,delta3,file=g)
  for i in range(n2):
    for j in range(n3):
      print(delta2*(i-n2/2),delta3*(j-n3/2),delta1*np.sum(Z[:,i,j]),file=g)
  g.close()  
  return

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
  arr=np.loadtxt(file_name,comments='#').reshape(n1,n2,-1)
  Z=arr[:,:,column]
  print('Maximum value = ',Z.max())
  print('Minimum value = ',Z.min())
  plt.figure()
  plt.imshow(Z,origin='lower',interpolation='nearest')
  plt.show()
  return
  
squared_displacement=np.zeros(nsteps+1)
tab_times=np.linspace(0.,t_max,num=nsteps+1)
density=np.zeros(L*L*L)

ir=0
filename='squared_displacement.dat'
f=open(filename,'w')

while(ir<nr):
  H=generate_hamiltonian(L,W)
  psir=generate_initial_state(L)
  Hpsir=np.tensordot(H,psir,axes=(0,0))
  energy=np.dot(psir,Hpsir)
#  print(energy)
#  if energy is not close to 0, reject it and generate a new configuration  
#  if (abs(energy)>50.0):
#    print("Energy =",energy," rejected!")
#    continue
  ir=ir+1
  print("Configuration",ir,"Energy=",energy)
  (energy_levels,eigenstates)=linalg.eigh(H) 
# The diagonalization routine does not sort the eigenvalues (which is stupid, by the way)
# Thus, sort them
  idx = np.argsort(energy_levels) 
  energy_levels=energy_levels[idx]
  eigenstates=eigenstates[:,idx]
  tab_coef=np.tensordot(eigenstates,psir,(0,0))
  for j in range(nsteps+1):
    time=t_step*j
    tab_cos=np.cos(time*energy_levels)
    tab_sin=np.sin(time*energy_levels)
    psi_t_r=np.tensordot(eigenstates,tab_cos*tab_coef,axes=(1,0))
    psi_t_i=np.tensordot(eigenstates,tab_sin*tab_coef,axes=(1,0))
#    energy=np.dot(psi_t_r,np.tensordot(H,psi_t_r,axes=(0,0)))+np.dot(psi_t_i,np.tensordot(H,psi_t_i,axes=(0,0)))
#    print(time,energy,compute_r2(psi_t_r)+compute_r2(psi_t_i))
    squared_displacement[j]+=compute_r2(psi_t_r)+compute_r2(psi_t_i)
  density[0:L*L*L]+=psi_t_r[0:L*L*L]**2+psi_t_i[0:L*L*L]**2
squared_displacement/=nr
filename='density.dat'
f=open(filename,'w')
f.write("# %d\n" % (L))
f.write("# %d\n" % (L))
f.write("# %d\n" % (L))
for i in range(L):
  for j in range(L):
    for k in range(L):
      f.write("%g\n" % (density[i*L*L+j*L+k]/nr))  
f.close()
extract_1d_and_2d_densities(filename)
view_density('density_xy.dat',block=False)
view_density('density_xz.dat',block=False)
view_density('density_yz.dat',block=False)
# Fit the squared displacement with a power law with additional background
# Known to be more or less correct in the critical regime with exponent 2/3
p0=[0.,1.,0.5]
p, var_matrix = curve_fit(power_law, tab_times[1:nsteps+1], squared_displacement[1:nsteps+1], p0=p0)
print("Best fit of squared displacement with power law + background: <r2(t)>=background+scaling_factor*t**exponent")
print()
print("background=",p[0],"scaling_factor=",p[1],"exponent=",p[2])
print()
filename_r2='squared_displacement.dat'
f=open(filename_r2,'w')
for j in range(nsteps+1):
  f.write("%g %g %g\n" % (tab_times[j],squared_displacement[j],power_law(tab_times[j],*p)))  
f.close()
print("Done, squared displacement (and fit) saved in",filename_r2)
print("Density saved in",filename)
print("Reduced densities in 1d and 2d are also saved") 
print("Hit Return to exit")   
input()  
      