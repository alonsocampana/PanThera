#!/bin/bash
  
for i in {1..10}
  do
	 python3 run_baseline.py --setting interpolation --fold $i --cpus 128 --model MuSyC --scale 
     python3 run_baseline.py --setting interpolation --fold $i --cpus 128 --model Zimmer --scale 
  done
  
for i in {1..10}
  do
	 python3 run_baseline.py --setting extrapolation --fold $i --cpus 128 --model MuSyC --scale 
     python3 run_baseline.py --setting extrapolation --fold $i --cpus 128 --model Zimmer --scale 
  done
  

