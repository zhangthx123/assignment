1. data_class.py
Defines all data structures required for SDVRP:
Representation of 4S store orders
Vehicle attributes (capacity, type, fixed cost)
Distance, time windows, and related parameters
Functions for loading and generating test instances
This module is imported by all algorithm files.

3. heuristic-NF.py
Implementation of the Sequential Next-Fit Algorithm (SNFA).

3. bb-random.py
Branch-and-bound solver with the BB_RANDOM branching rule.

4. bb-rv-shunxu.py
Branch-and-bound solver with deterministic branching strategies:
Implements BB_RV_SMALL and/or BB_RV_LARGE:
RV_SMALL: choose vehicle with smallest remaining volume
RV_LARGE: choose vehicle with largest remaining volume.

5. mymodel_cplex.py
The MILP baseline model using IBM CPLEX.

6. data/ (folder)
Contains all datasets used for computational experiments:
Randomly generated SDVRP instances
Different problem sizes (5–20 orders)
Format compatible with data_class.py
