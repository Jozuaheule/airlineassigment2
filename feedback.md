# Feedback on Airline Assignment Code

This document provides an assessment of the `airline_assignment.py` script against the requirements outlined in `Assignment2_2025.pdf`. The analysis focuses on the static demand model.

The code implements a dynamic programming approach to solve the aircraft routing and scheduling problem. While the overall structure is sound and correctly follows many of the assignment's specifications (e.g., time discretization, flight time calculation, basic constraints), there are four key areas where the implementation deviates from the requirements.

## 1. Hub-and-Spoke Constraint Not Enforced

**Observation:** The current model allows for the creation of point-to-point routes (e.g., a flight from a spoke airport to another spoke airport). The DP formulation considers flying from the current airport `i` to any other airport `j`.

**Requirement:** The assignment explicitly states, "...only flights to and from the hub are considered."

**Impact:** The model generates schedules that are not valid under the assignment's rules, potentially finding profitable but disallowed routes.

**Suggestion:** Modify the DP logic. When an aircraft is at a spoke airport (any airport other than the hub, "EHAM"), the only permissible flying decision should be to return to the hub. When at the hub, it can fly to any spoke. This simplifies the decision space at each state.

---

## 2. Incorrect Demand Depletion Model

**Observation:** The `_reconstruct_and_deplete_demand` function uses a simplified, proportional approach to reduce demand. It calculates the fraction of passengers taken from the total available demand in the three-hour window and reduces the *total daily demand* by that fraction.

**Requirement:** The assignment specifies a sequential depletion mechanism: "After adding an aircraft route to your solution, you should remove the demand that you have transported. To do this, remove the demand from t, t − 1, and t - 2, sequential, until the aircraft is full or no more demand is available."

**Impact:** The current method inaccurately depletes demand. By reducing the overall daily demand, it affects future calculations for all time slots, not just the specific hourly buckets that were drawn from. This will lead to incorrect passenger numbers for subsequent flights.

**Suggestion:** The demand model needs to be more granular. Instead of a single daily demand value per route, you should maintain an array or list of 24 hourly demand values for each route. When a flight occurs, you would attempt to fill the aircraft by sequentially drawing passengers from the demand buckets for hour `t`, then `t-1`, then `t-2`, and reducing those specific buckets to zero as they are consumed.

---

## 3. Daily Lease Cost Is Omitted from Profit Calculation

**Observation:** The `calculate_costs` function correctly computes the operating costs per flight leg (fixed, time-based, and fuel). However, the daily lease cost for each aircraft, while loaded from the `FleetType.xlsx` file, is never used to evaluate the profitability of a full-day schedule. The DP's value function, `V`, maximizes the sum of per-leg profits.

**Requirement:** "All aircraft are leased, and therefore a leasing cost needs to be accounted for." And "You do not have to use all the aircraft in your fleet. Only if it is profitable."

**Impact:** The model may generate and keep schedules for aircraft where the total profit from all flights in a day does not exceed the daily lease cost, making the use of that aircraft a net loss.

**Suggestion:** After reconstructing the full schedule for an aircraft (`_reconstruct_and_deplete_demand`), calculate the total profit for that schedule. Compare this total profit against the aircraft's `lease_cost_eur_day`. If the profit is less than the lease cost, the entire schedule for that aircraft should be discarded.

---

## 4. Load Factor Assumption Misinterpreted

**Observation:** The code calculates the number of passengers for a flight by taking the minimum of available seats and the actual available demand (`passengers = min(aircraft_specs.get('Seats', 0), sum(demands))`).

**Requirement:** "Assume a load factor of 80%."

**Impact:** This is a subtle point of interpretation. The requirement seems to be a simplifying assumption for the planning model. By simulating the exact number of passengers, the model is more complex than requested. A strict interpretation would mean revenue should be calculated based on the assumption that an aircraft will be filled to 80% capacity if sufficient demand exists.

**Suggestion:** Change the passenger calculation. The number of passengers on a flight should be `min(sum(demands), 0.80 * aircraft_specs.get('Seats', 0))`. This aligns with the idea of planning for an 80% load factor while still being limited by available demand. Ensure that the demand that will be removed is 0,8 as well.