import numpy as np
import pandas as pd
import math
from data_processing import *

class DynamicProgrammingModel:
    """
    A dynamic programming model to solve the aircraft routing and scheduling problem.
    This implementation uses a greedy heuristic, solving the schedule for one aircraft
    at a time and depleting demand, which is a practical approach for this complex
    resource allocation problem.
    """

    def __init__(self, airports_df, demand_df, fleet_df, hour_coefficients_df, distance_matrix_df):
        self.airports = airports_df
        self.original_demand = demand_df.copy()
        self.fleet_data = fleet_df
        self.hour_coefficients = hour_coefficients_df
        self.distance_matrix = distance_matrix_df
        
        # --- Model Constants ---
        self.TIME_STEP_MINUTES = 6
        self.N_T = int(24 * 60 / self.TIME_STEP_MINUTES)
        self.HUB_AIRPORT_ICAO = "EHAM"
        self.MIN_BLOCK_HOURS = 6
        self.LOAD_FACTOR = 0.8

        # --- Data Mappings ---
        self.airport_icaos = self.airports.index.tolist()
        self.icao_to_idx = {icao: i for i, icao in enumerate(self.airport_icaos)}
        self.idx_to_icao = {i: icao for i, icao in enumerate(self.airport_icaos)}
        self.N_P = len(self.airport_icaos)
        self.N_A = len(self.fleet_data)

        # --- Initialize Hourly Demand ---
        self.hourly_demand = np.zeros((self.N_P, self.N_P, 24))
        for i in range(self.N_P):
            for j in range(self.N_P):
                if i == j: continue
                origin_icao = self.idx_to_icao[i]
                dest_icao = self.idx_to_icao[j]
                if origin_icao in self.original_demand.index and dest_icao in self.original_demand.columns:
                    daily_demand = self.original_demand.loc[origin_icao, dest_icao]
                    if origin_icao in self.hour_coefficients.index:
                        for hour in range(24):
                            coefficient = self.hour_coefficients.loc[origin_icao, hour]
                            self.hourly_demand[i, j, hour] = daily_demand * coefficient

    def _get_demand(self, time_step, origin_idx, dest_idx, current_hourly_demand):
        """Calculates demand for the three relevant hourly buckets."""
        demands = []
        hour_t = int(np.floor(time_step / (60 / self.TIME_STEP_MINUTES)))
        for t_offset in range(3):  # For hours t, t-1, t-2
            hour = hour_t - t_offset
            if hour < 0:
                demands.append(0)
                continue
            demands.append(current_hourly_demand[origin_idx, dest_idx, hour])
        return demands

    def _reconstruct_and_deplete_demand(self, k, D, modifiable_hourly_demand):
        """
        Reconstructs the schedule for a single aircraft, calculates its profitability,
        and depletes demand sequentially.
        """
        schedule = []
        block_time_minutes = 0
        total_profit = 0
        current_t = 0
        current_loc_idx = self.icao_to_idx[self.HUB_AIRPORT_ICAO]
        aircraft_specs = self.fleet_data.iloc[k]

        while current_t < self.N_T:
            decision_idx = D[current_t, current_loc_idx]
            if decision_idx == -1 or decision_idx == current_loc_idx:
                current_t += 1
                continue

            origin_icao = self.idx_to_icao[current_loc_idx]
            dest_icao = self.idx_to_icao[decision_idx]
            distance = self.distance_matrix.loc[origin_icao, dest_icao]
            speed = aircraft_specs.get('speed_kmh', 0)
            
            flight_duration_minutes = (distance / speed * 60) + 30 if speed > 0 else float('inf')
            flight_duration_steps = int(np.ceil(flight_duration_minutes / self.TIME_STEP_MINUTES))

            # --- Sequential Demand Depletion Logic ---
            hour_t = int(np.floor(current_t / (60 / self.TIME_STEP_MINUTES)))
            passengers_to_take = self.LOAD_FACTOR * aircraft_specs.get('seats', 0)
            passengers_taken = 0
            
            demands_to_deplete = []
            for t_offset in range(3):
                if passengers_taken >= passengers_to_take: break
                hour = hour_t - t_offset
                if hour < 0: continue
                
                available_demand_in_hour = modifiable_hourly_demand[current_loc_idx, decision_idx, hour]
                can_take = min(passengers_to_take - passengers_taken, available_demand_in_hour)
                
                if can_take > 0:
                    passengers_taken += can_take
                    demands_to_deplete.append({'hour': hour, 'amount': can_take})

            if passengers_taken > 0:
                for item in demands_to_deplete:
                    modifiable_hourly_demand[current_loc_idx, decision_idx, item['hour']] -= item['amount']
                
                revenue = calculate_revenue(distance, passengers_taken)
                costs = calculate_costs(distance, aircraft_specs)['total_cost']
                profit = revenue - costs
                total_profit += profit

                schedule.append({
                    "aircraft_id": k, "type": aircraft_specs.name,
                    "origin": origin_icao, "destination": dest_icao,
                    "departure_time": current_t, "arrival_time": current_t + flight_duration_steps,
                    "passengers": round(passengers_taken)
                })

            block_time_minutes += flight_duration_minutes
            current_t += flight_duration_steps
            current_loc_idx = decision_idx

        if block_time_minutes / 60 >= self.MIN_BLOCK_HOURS:
            return schedule, block_time_minutes, total_profit
        
        return None, 0, 0

    def solve(self):
        """
        Implements the full DP solver with a greedy heuristic for demand allocation.
        """
        final_schedules = []
        modifiable_hourly_demand = self.hourly_demand.copy()

        print("Solving for each aircraft sequentially...")
        for k in range(self.N_A):
            aircraft_specs = self.fleet_data.iloc[k]
            print(f"  Aircraft {k} ({aircraft_specs.name})...")

            V = np.zeros((self.N_T, self.N_P))
            D = np.full((self.N_T, self.N_P), -1, dtype=int)

            for t in range(self.N_T - 2, -1, -1):
                for i in range(self.N_P):
                    v_stay = V[t + 1, i]
                    best_value = v_stay
                    best_decision = i

                    hub_idx = self.icao_to_idx[self.HUB_AIRPORT_ICAO]
                    possible_destinations = [hub_idx] if i != hub_idx else list(range(self.N_P))

                    for j in possible_destinations:
                        if i == j: continue
                        
                        origin_icao = self.idx_to_icao[i]
                        dest_icao = self.idx_to_icao[j]
                        distance = self.distance_matrix.loc[origin_icao, dest_icao]

                        if distance > aircraft_specs.get('range_km', 0): continue
                        if self.airports.loc[dest_icao].get('Runway length (m)', 0) < aircraft_specs.get('runway_m', 0): continue
                        if self.airports.loc[origin_icao].get('Runway length (m)', 0) < aircraft_specs.get('runway_m', 0): continue

                        speed = aircraft_specs.get('speed_kmh', 0)
                        flight_duration_minutes = (distance / speed * 60) + 30 if speed > 0 else float('inf')
                        flight_duration_steps = int(np.ceil(flight_duration_minutes / self.TIME_STEP_MINUTES))
                        t_arrival = t + flight_duration_steps

                        if t_arrival >= self.N_T: continue

                        demands = self._get_demand(t, i, j, modifiable_hourly_demand)
                        passengers = min(sum(demands), self.LOAD_FACTOR * aircraft_specs.get('seats', 0))
                        
                        if passengers <= 0: continue

                        revenue = calculate_revenue(distance, passengers)
                        costs = calculate_costs(distance, aircraft_specs)['total_cost']
                        profit = revenue - costs

                        v_fly = profit + V[t_arrival, j]

                        if v_fly > best_value:
                            best_value = v_fly
                            best_decision = j

                    V[t, i] = best_value
                    D[t, i] = best_decision

            # --- Forward Pass, Profitability Check & Demand Depletion ---
            temp_hourly_demand = modifiable_hourly_demand.copy()
            schedule, block_time, total_profit = self._reconstruct_and_deplete_demand(k, D, temp_hourly_demand)
            
            lease_cost = aircraft_specs.get('lease_cost_eur_day', 0)
            if schedule and (total_profit > lease_cost):
                print(f"    -> Profitable schedule found: {block_time/60:.2f} block hours, Profit (after lease): {total_profit - lease_cost:.2f} €")
                final_schedules.extend(schedule)
                modifiable_hourly_demand = temp_hourly_demand # Commit demand depletion
            else:
                print(f"    -> Schedule not profitable enough. Profit: {total_profit:.2f}, Lease Cost: {lease_cost:.2f}. Discarding.")
        
        return final_schedules

def calculate_revenue(distance, passengers):
    """Calculates revenue based on RPK."""
    if distance <= 0 or passengers <= 0: return 0
    yield_per_rpk = 5.9 * (distance ** -0.76) + 0.043
    return yield_per_rpk * distance * passengers

def calculate_costs(distance, aircraft_type_specs):
    """Calculates operating costs for a flight leg."""
    fixed = aircraft_type_specs.get('fixed_op_cost_eur_flight', 0)
    speed = aircraft_type_specs.get('speed_kmh', 0)
    flight_hours = distance / speed if speed > 0 else 0
    time_based = aircraft_type_specs.get('time_based_cost_eur_hour', 0) * flight_hours
    fuel = aircraft_type_specs.get('fuel_cost_eur_kg', 0) * (distance / 1.5)
    total = fixed + time_based + fuel
    return {'total_cost': total, 'fixed': fixed, 'time': time_based, 'fuel': fuel}

def main():
    """Main function to load data, run the model, and print results."""
    print("--- Loading All Data ---")
    airports, demand = load_demand_and_airport_data()
    fleet = load_fleet_data()
    hour_coeffs = load_hour_coefficients()
    distance_matrix = calculate_distance_matrix(airports)
    print("--- Data Loading Complete ---\n")

    dp_model = DynamicProgrammingModel(
        airports_df=airports,
        demand_df=demand,
        fleet_df=fleet,
        hour_coefficients_df=hour_coeffs,
        distance_matrix_df=distance_matrix
    )

    optimal_schedule = dp_model.solve()

    print("\n--- Optimal Schedule Found ---")
    if not optimal_schedule:
        print("No profitable flights found.")
    else:
        schedule_df = pd.DataFrame(optimal_schedule)
        # Convert time steps to HH:MM format for readability
        schedule_df['departure_time'] = schedule_df['departure_time'].apply(
            lambda ts: f"{int(ts*6/60):02d}:{int(ts*6%60):02d}"
        )
        schedule_df['arrival_time'] = schedule_df['arrival_time'].apply(
            lambda ts: f"{int(ts*6/60):02d}:{int(ts*6%60):02d}"
        )
        print(schedule_df.to_string())

if __name__ == "__main__":
    main()

