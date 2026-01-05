# Analysis of Complexity Levels in the Airline Assignment Problem

Based on the assignment description and the structure of the provided Python code, the aircraft routing and scheduling problem can be broken down into three distinct levels of complexity. Each level builds upon the previous one, adding new layers of constraints and decision variables.

## Level 1: Single Flight Leg Economics

This is the most fundamental level of the problem. It focuses on the costs and revenues associated with a single, one-way flight leg between the hub and another airport. This level does not consider the scheduling aspect (time of day) or the assignment of specific, individual aircraft.

### Key Components:
- **Cost Calculation:** As detailed in Appendix C, this involves calculating the total operating costs for a given aircraft type on a given route. This includes:
    - **Fixed Operating Costs:** Costs per flight leg (e.g., landing fees).
    - **Time-Based Costs:** Costs related to flight duration (e.g., crew salary).
    - **Fuel Costs:** Costs dependent on flight distance.
- **Revenue Calculation:** As described in Appendix B, this involves calculating the potential revenue from a single flight, assuming a certain load factor (80%). Revenue is based on a yield formula that is dependent on the flight distance.
- **Basic Constraints:** This level also incorporates the most basic operational constraints, such as ensuring an aircraft type has sufficient range to fly a route and that the destination airport's runway is long enough.

At this level, one can determine the profitability of using a specific aircraft type on a specific route, independent of time.

## Level 2: Time-Dependent Scheduling and Demand

This level introduces the dimension of time into the problem, moving from a static analysis of a single leg to a dynamic scheduling problem over a 24-hour period.

### Key Components:
- **Time Discretization:** The scheduling horizon is divided into discrete time steps (6-minute intervals). All decisions (departing, staying at an airport) are made at these steps.
- **Hourly Demand:** Demand is not constant throughout the day. This level incorporates the `HourCoefficients.xlsx` data to model how demand for a route fluctuates hour by hour (as per Appendix A).
- **Demand Capture:** The model must account for the rule that a flight departing at a certain time can capture demand from the current hour and the two preceding hours. This introduces a time-dependent element to revenue calculation.
- **Dynamic Programming Framework:** The problem is structured as a dynamic program, where "stages" are time steps and "states" likely represent an aircraft's location. The decisions at each stage involve choosing the next flight.

This level transforms the problem from "is this route profitable?" to "when is the most profitable time to fly this route?".

## Level 3: Fleet Assignment and Full Network Optimization

This is the highest and most complex level, where the goal is to create a complete, optimal schedule for the entire fleet of aircraft to maximize total profit. It integrates the first two levels and adds network-level constraints and interdependencies.

### Key Components:
- **Fleet as a Whole:** Decisions are no longer about a single, generic aircraft. The model must assign specific aircraft from the available fleet to specific sequences of flights (routes).
- **Aircraft State Tracking:** The model must track the state (time and location) of every individual aircraft in the fleet throughout the 24-hour period.
- **Global Constraints:** This level enforces constraints that apply to each aircraft's full-day schedule, such as:
    - Each aircraft must start and end its day at the hub.
    - Each aircraft utilized must meet a minimum of 6 hours of block time.
- **Demand Depletion (Spill/Recapture):** This is a critical network effect. When one flight carries passengers, that demand is removed and is no longer available for other potential flights on the same route. The model must manage the "pool" of available demand as it is consumed by scheduled flights.
- **Optimal Policy:** The final output is not just a single decision, but a complete policy—an optimal schedule of flights for each aircraft in the fleet that maximizes the airline's total profit over the 24-hour period.
