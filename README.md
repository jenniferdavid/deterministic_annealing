This repo contains:
1. Deterministic Annealing (potts_spin.cpp)
2. Simulated Annealing 
3. Random Search
4. Brute Force (solutions.cpp)

INPUT:
Deterministic Annealing:
./potts_spin 2 3 -random -max 100 0.01 10 100
./potts_spin $no_of_vehicles$ $no_of_tasks$ $-random_or_-read$ $-max_or_-sum$ $kT_start$ $kT_end$ $kT_fac$ $gamma$

Simulated Annealing:
./SA 2 3 -random -max 100 0.01 10 100

Brute Force:
./solutions 2 3 -random -max
./solutions $no_of_vehicles$ $no_of_tasks$ $-random_or_-read$ $-max_or_-sum$

Random Search
./RS 2 3 -random $no_of_iterations$

OTHER OPTIONS:
-random or -read
-max or -sum
