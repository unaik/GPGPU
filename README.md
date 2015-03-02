# GPGPU - General purpose computing using a graphics processing unit


1. Lattice Boltzmann Water Waves:
Simulated ocean waves and reflections of ocean waves off rigid obstructions as described in paper :Lattice Boltzmann Water Waves by Robert Geist, et al.
In addition, surface normal calculations, wave height calculations and surface color calculations were also performed. 
Calculations for all parameters of 8 waves were performed parallelly on the GPU. 
Implemented in C using OpenGL for rendering and OpenCL for GPU computing.

2. Particle system: 
A particle system which calculates  velocity verlet update and bounce(off 2 surfaces) for more than a million particles. 
All the calculations are performed  parallelly on the GPU. 
System was written in C using OpenGL for rendering and OpenCL for GPU computing.
