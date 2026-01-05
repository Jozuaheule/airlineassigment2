# Implementation Plan for the Dynamic Programming Model

This document outlines the plan for implementing the dynamic programming logic in the `solve` method of the `DynamicProgrammingModel` class. The implementation will follow the problem description and build upon the three levels of complexity identified previously.

## 1. Data Initialization and Pre-processing

Before starting the main DP loop, several data structures need to be initialized.

- **Time Steps:** Create a list or array representing the time steps for the 24-hour horizon (24 hours * 60 min/hour / 6 min/step = 240 steps).
- **Airport Data:** Load and pre-process airport data. This includes calculating the distance matrix between the hub and all other airports. The hub airport needs to be identified.
- **Fleet Data:** Load and pre-process fleet data.
- **Demand Data:** Load the hourly demand data. It should be structured for easy lookup by `(origin, destination, hour)`.
- **DP Value Function Table:** Initialize a data structure to store the value function `V(k, t, i)`, which will represent the maximum profit achievable starting from time `t` at airport `i` with aircraft `k`. A multi-dimensional array or a dictionary of dictionaries could be used: `V[aircraft_id][time_step][airport_id]`. Initialize all values to zero or a very small number.
- **Policy/Decision Table:** Initialize a parallel data structure `D(k, t, i)`. Crucially, this table will only store the single, optimal *next action* (e.g., the destination airport for a flight, or a special marker for "stay") for each state. It does not store the entire history. This table is used after the main loop to reconstruct the final schedule by chaining these individual optimal decisions together.

## 2. The Core Dynamic Programming Loop

The `solve` method will implement the backward recursion of the dynamic programming model.

- **Iterate through time steps `t` in reverse order**, from the last step (t=239) to the first (t=0).
- **Iterate through each aircraft `k`** in the fleet.
- **Iterate through each possible state (airport `i`)** where an aircraft could be located at time `t`.

### Inside the loop (for each `(t, k, i)`):

1.  **Evaluate the "Stay" Option:**
    - The value of staying at the current airport `i` until the next time step.
    - `V_stay = V(k, t+1, i)`.
    - This will be one of the candidate values for `V(k, t, i)`.

2.  **Evaluate "Fly" Options:**
    - For each possible destination airport `j` (from `i`):
        - Calculate the flight time, including extra time for take-off and landing.
        - Determine the arrival time step `t_arrival`.
        - If `t_arrival` is within the 24-hour horizon:
            - Calculate the revenue for the flight from `i` to `j` using the implemented `calculate_revenue` function. This requires looking up the demand for the relevant hours.
            - Calculate the costs for the flight using `calculate_costs`.
            - Calculate the immediate profit of this flight: `profit_ij = revenue - costs`.
            - Look up the future value from the DP table at the arrival state: `V_future = V(k, t_arrival, j)`.
            - The total value for this "fly" option is `V_fly_ij = profit_ij + V_future`.

3.  **Update the Value Function:**
    - Compare the value of all "fly" options and the "stay" option.
    - Set `V(k, t, i)` to the maximum value found.
    - Store the decision that led to this maximum value in the policy table `D(k, t, i)`.

## 3. Handling Demand and Fleet Constraints

- **Demand Depletion:** This is a key complexity. The available demand for a route at a given hour is a shared resource.
    - **Proposed Solution:** The state of the DP model needs to be expanded to include the remaining demand. This would drastically increase complexity.
    - **Alternative (Heuristic) Approach:** A simpler, more tractable approach is to iterate through the aircraft one by one. For each aircraft, solve the DP model assuming all demand is available. After finding the schedule for one aircraft, "remove" the demand it has captured from the total demand pool. Then, solve the DP for the next aircraft with the reduced demand. This is a greedy approach but is computationally more feasible.
- **Aircraft Constraints:**
    - **Start and End at Hub:** The initial state at `t=0` for all aircraft is the hub. The final value function should reflect the value of ending at the hub.
    - **Minimum Block Time:** This can be checked after the full schedule for an aircraft is reconstructed. If an aircraft's schedule doesn't meet the minimum block time, its schedule can be discarded.

## 4. Reconstructing the Optimal Schedule

After the DP table has been filled (the loops are complete), the final step is to reconstruct the optimal schedule.

- For each aircraft `k`, start at `t=0` at the hub airport.
- Use the policy table `D(k, t, i)` to trace the sequence of decisions (flights) forward in time.
- Follow the decisions from `t=0` to the end of the horizon, building the flight schedule for each aircraft.
- Store the results in a clear, readable format (e.g., a list of flight objects or a pandas DataFrame).

This plan provides a structured approach to building the complex logic required for the dynamic programming solver. The heuristic for demand depletion is a key simplification to make the problem tractable.
