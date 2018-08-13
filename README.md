This repo contains:
1. Deterministic Annealing (potts_spin.cpp)
2. Simulated Annealing 
3. Random Search
4. Brute Force (solutions.cpp)

INPUT:
Deterministic Annealing: (9 arguments)
./potts_spin 2 3 -random -max 100 0.01 10 100
./potts_spin $no_of_vehicles$ $no_of_tasks$ $-random_or_-read$ $-max_or_-sum$ $kT_start$ $kT_end$ $kT_fac$ $gamma$

Simulated Annealing: (8 arguments)
./SA 2 3 -random -max 100 0.01 10
./SA $no_of_vehicles$ $no_of_tasks$ $-random_or_-read$ $-max_or_-sum$ $kT_start$ $kT_end$ $kT_fac$ 

Brute Force: (5 arguments)
./solutions 2 3 -random -max
./solutions $no_of_vehicles$ $no_of_tasks$ $-random_or_-read$ $-max_or_-sum$

Random Search: (6 arguments)
./RS 2 3 -random -max $no_of_iterations$

OTHER OPTIONS:
-random or -read
-max or -sum
