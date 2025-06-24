import pandas as pd
import numpy as np
from datetime import datetime
from geopy.distance import geodesic
import matplotlib.pyplot as plt

def plot_comprehensive_trip_analysis(trips):
    """
    Comprehensive visualization combining multiple perspectives of trip data.
    """
    if not trips:
        print("No trips to plot!")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Trip ID vs Data Point ID (scatter)
    trip_ids = []
    data_point_ids = []
    
    for trip_idx, trip in enumerate(trips):
        trip_id = trip_idx + 1
        for point_idx in range(len(trip)):
            trip_ids.append(trip_id)
            data_point_ids.append(point_idx + 1)
    
    ax1.scatter(data_point_ids, trip_ids, alpha=0.6, s=20, c='blue')
    ax1.set_xlabel('Data Point ID (within trip)')
    ax1.set_ylabel('Trip ID')
    ax1.set_title('Trip ID vs Data Point ID')
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks(range(1, len(trips) + 1))
    
    # Plot 2: Cumulative data points across trips
    cumulative_points = []
    trip_boundaries = [0]
    
    for trip in trips:
        if cumulative_points:
            cumulative_points.extend(range(cumulative_points[-1] + 1, cumulative_points[-1] + len(trip) + 1))
        else:
            cumulative_points.extend(range(1, len(trip) + 1))
        trip_boundaries.append(cumulative_points[-1])
    
    trip_ids_cumulative = []
    for trip_idx, trip in enumerate(trips):
        trip_ids_cumulative.extend([trip_idx + 1] * len(trip))
    
    ax2.plot(cumulative_points, trip_ids_cumulative, 'o-', markersize=3, alpha=0.7)
    for boundary in trip_boundaries[1:-1]:  # Skip first (0) and last
        ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Cumulative Data Point ID')
    ax2.set_ylabel('Trip ID')
    ax2.set_title('Cumulative Data Points Across All Trips')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Data points per trip (bar chart)
    trip_nums = list(range(1, len(trips) + 1))
    point_counts = [len(trip) for trip in trips]
    
    bars = ax3.bar(trip_nums, point_counts, alpha=0.7, color='green')
    ax3.set_xlabel('Trip ID')
    ax3.set_ylabel('Number of Data Points')
    ax3.set_title('Data Points per Trip')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trip duration vs data points
    durations = []
    for trip in trips:
        start_time = trip['Data Actual Time'].iloc[0]
        end_time = trip['Data Actual Time'].iloc[-1]
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        durations.append(duration)
    
    ax4.scatter(point_counts, durations, alpha=0.7, s=50, c='purple')
    ax4.set_xlabel('Number of Data Points')
    ax4.set_ylabel('Trip Duration (minutes)')
    ax4.set_title('Trip Duration vs Data Points')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(point_counts, durations)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def extract_trips_from_gps_data(df, proximity_threshold=0.1, min_trip_duration_minutes=30, min_trip_points=10):
    """
    Extract circular trips from GPS data where trips start and end at similar locations.
    
    Parameters:
    - df: DataFrame with GPS data
    - proximity_threshold: Distance threshold in km to consider two points as "same location"
    - min_trip_duration_minutes: Minimum duration for a valid trip
    - min_trip_points: Minimum number of GPS points for a valid trip
    
    Returns:
    - List of DataFrames, each representing a trip
    """
    
    # Ensure the data is sorted by time
    df_sorted = df.copy()
    df_sorted['Data Actual Time'] = pd.to_datetime(df_sorted['Data Actual Time'], format='%d-%m-%Y %I:%M:%S %p')
    df_sorted = df_sorted.sort_values('Data Actual Time').reset_index(drop=True)
    
    # Add date column for daily grouping
    df_sorted['Date'] = df_sorted['Data Actual Time'].dt.date
    
    trips = []
    
    # Process each day separately
    for date in df_sorted['Date'].unique():
        daily_data = df_sorted[df_sorted['Date'] == date].copy().reset_index(drop=True)
        
        if len(daily_data) < min_trip_points:
            continue
            
        # Find potential trip start/end points
        trip_starts = []
        current_trip_start = 0
        
        for i in range(1, len(daily_data)):
            # Check if we've returned to starting location
            start_lat = daily_data.iloc[current_trip_start]['Latitude']
            start_lon = daily_data.iloc[current_trip_start]['Longitude']
            current_lat = daily_data.iloc[i]['Latitude']
            current_lon = daily_data.iloc[i]['Longitude']
            
            # Calculate distance between start and current point
            distance = geodesic((start_lat, start_lon), (current_lat, current_lon)).kilometers
            
            # If we're back near the starting point
            if distance <= proximity_threshold:
                # Check if trip duration and point count meet minimum requirements
                trip_duration = (daily_data.iloc[i]['Data Actual Time'] - 
                               daily_data.iloc[current_trip_start]['Data Actual Time']).total_seconds() / 60
                
                if trip_duration >= min_trip_duration_minutes and (i - current_trip_start + 1) >= min_trip_points:
                    # Extract the trip
                    trip_data = daily_data.iloc[current_trip_start:i+1].copy()
                    trip_data['Trip_ID'] = f"Trip_{date}_{len(trips)+1}"
                    trips.append(trip_data)
                    
                    # Start looking for next trip
                    current_trip_start = i
                    
    return trips

def analyze_trips(trips):
    """
    Analyze extracted trips and provide summary statistics.
    """
    if not trips:
        print("No trips found!")
        return
    
    print(f"Total trips extracted: {len(trips)}")
    print("\nTrip Analysis:")
    print("-" * 50)
    
    for i, trip in enumerate(trips):
        trip_id = trip['Trip_ID'].iloc[0]
        start_time = trip['Data Actual Time'].iloc[0]
        end_time = trip['Data Actual Time'].iloc[-1]
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        # Calculate trip distance (approximate)
        total_distance = 0
        for j in range(1, len(trip)):
            lat1, lon1 = trip.iloc[j-1]['Latitude'], trip.iloc[j-1]['Longitude']
            lat2, lon2 = trip.iloc[j]['Latitude'], trip.iloc[j]['Longitude']
            total_distance += geodesic((lat1, lon1), (lat2, lon2)).kilometers
        
        print(f"{trip_id}:")
        print(f"  Start: {start_time}")
        print(f"  End: {end_time}")
        print(f"  Duration: {duration:.1f} minutes")
        print(f"  Points: {len(trip)}")
        print(f"  Approximate distance: {total_distance:.2f} km")
        
        # Check circularity
        start_lat, start_lon = trip.iloc[0]['Latitude'], trip.iloc[0]['Longitude']
        end_lat, end_lon = trip.iloc[-1]['Latitude'], trip.iloc[-1]['Longitude']
        circularity_distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers
        print(f"  Start-End distance: {circularity_distance:.3f} km")
        print()

def plot_trips(trips, max_trips_to_plot=5):
    """
    Plot the extracted trips on a map-like visualization.
    """
    if not trips:
        print("No trips to plot!")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot up to max_trips_to_plot trips
    for i, trip in enumerate(trips[:max_trips_to_plot]):
        trip_id = trip['Trip_ID'].iloc[0]
        plt.plot(trip['Longitude'], trip['Latitude'], 'o-', 
                label=trip_id, alpha=0.7, markersize=3)
        
        # Mark start and end points
        plt.plot(trip['Longitude'].iloc[0], trip['Latitude'].iloc[0], 
                's', markersize=8, color='green', alpha=0.8)
        plt.plot(trip['Longitude'].iloc[-1], trip['Latitude'].iloc[-1], 
                '^', markersize=8, color='red', alpha=0.8)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Extracted Bus Trips in KTM Valley')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_trip_from_index(trips, index=0):
    """
    Plot the extracted trips on a map-like visualization.
    """
    if not trips:
        print("No trips to plot!")
        return
    
    plt.figure(figsize=(12, 8))

    trip = trips[index]
    
    # Plot up to max_trips_to_plot trips
    # for i, trip in enumerate(trips[index]):
    trip_id = trip['Trip_ID'].iloc[0]
    plt.plot(trip['Longitude'], trip['Latitude'], 'o-', 
            label=trip_id, alpha=0.7, markersize=3)
        
    # Mark start and end points
    plt.plot(trip['Longitude'].iloc[0], trip['Latitude'].iloc[0], 
            's', markersize=8, color='green', alpha=0.8)
    plt.plot(trip['Longitude'].iloc[-1], trip['Latitude'].iloc[-1], 
            '^', markersize=8, color='red', alpha=0.8)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Extracted Bus Trips in KTM Valley')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




def plot_trip_vs_datapoint(trips, original_df=None):
    """
    Plot Trip ID vs Actual Data Point ID graph showing the distribution of data points across trips.
    This uses the original index from the dataset to show which data points are NOT in any trip.
    
    Parameters:
    - trips: List of trip DataFrames
    - original_df: Original DataFrame to get the full range of data point IDs
    """
    if not trips:
        print("No trips to plot!")
        return
    
    # Prepare data for plotting
    trip_ids = []
    actual_data_point_ids = []
    
    for trip_idx, trip in enumerate(trips):
        trip_id = trip_idx + 1  # Use numeric trip ID for plotting
        # Get the original indices from the trip data
        for original_idx in trip.index:
            trip_ids.append(trip_id)
            actual_data_point_ids.append(original_idx)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # If original_df is provided, show the full context
    if original_df is not None:
        # Plot all data points in light gray to show the full dataset
        all_indices = list(original_df.index)
        plt.scatter(all_indices, [0] * len(all_indices), alpha=0.3, s=15, c='lightgray', 
                   label='Data points NOT in trips', marker='_')
    
    # Scatter plot for trip data points
    colors = plt.cm.tab10(np.linspace(0, 1, len(trips)))  # Different colors for each trip
    
    for trip_idx, trip in enumerate(trips):
        trip_id = trip_idx + 1
        trip_indices = list(trip.index)
        trip_ids_for_this_trip = [trip_id] * len(trip_indices)
        
        plt.scatter(trip_indices, trip_ids_for_this_trip, alpha=0.7, s=40, 
                   c=[colors[trip_idx]], label=f'Trip {trip_id}', 
                   edgecolors='black', linewidth=0.5)
    
    # Customize the plot
    plt.xlabel('Actual Data Point ID (Original Dataset Index)', fontsize=12)
    plt.ylabel('Trip ID', fontsize=12)
    plt.title('Trip ID vs Actual Data Point ID Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to show all trip IDs plus 0 for non-trip points
    if original_df is not None:
        plt.yticks(list(range(0, len(trips) + 1)))
        plt.ylabel('Trip ID (0 = Not in any trip)', fontsize=12)
    else:
        plt.yticks(range(1, len(trips) + 1))
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add statistics
    total_trip_points = len(actual_data_point_ids)
    total_original_points = len(original_df) if original_df is not None else total_trip_points
    unused_points = total_original_points - total_trip_points
    avg_points_per_trip = total_trip_points / len(trips) if trips else 0
    
    stats_text = f'Total Trips: {len(trips)}\n'
    stats_text += f'Points in Trips: {total_trip_points}\n'
    if original_df is not None:
        stats_text += f'Total Dataset Points: {total_original_points}\n'
        stats_text += f'Unused Points: {unused_points} ({unused_points/total_original_points*100:.1f}%)\n'
    stats_text += f'Avg Points/Trip: {avg_points_per_trip:.1f}'
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis of unused points
    if original_df is not None and unused_points > 0:
        used_indices = set(actual_data_point_ids)
        unused_indices = [idx for idx in original_df.index if idx not in used_indices]
        
        print(f"\nDetailed Analysis of Unused Data Points:")
        print(f"Total unused points: {len(unused_indices)}")
        # print(f"Unused indices: {sorted(unused_indices)}")
        
        if len(unused_indices) <= 20:  # Show details for small numbers
            print("\nUnused data points details:")
            for idx in sorted(unused_indices):
                row = original_df.loc[idx]
                print(f"Index {idx}: {row['Actual Time Data String']} - "
                      f"Lat: {row['Latitude']:.6f}, Lon: {row['Longitude']:.6f}")
        else:
            print(f"\nFirst 10 unused indices: {sorted(unused_indices)[:10]}")
            print(f"Last 10 unused indices: {sorted(unused_indices)[-10:]}")



def plot_trip_timeline(trips):
    """
    Alternative visualization: Plot trips as horizontal bars showing their timeline and data point distribution.
    """
    if not trips:
        print("No trips to plot!")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Trip timeline
    for trip_idx, trip in enumerate(trips):
        trip_id = trip_idx + 1
        start_time = trip['Data Actual Time'].iloc[0]
        end_time = trip['Data Actual Time'].iloc[-1]
        
        # Convert to hours for plotting
        start_hour = start_time.hour + start_time.minute/60
        end_hour = end_time.hour + end_time.minute/60
        
        ax1.barh(trip_id, end_hour - start_hour, left=start_hour, height=0.6, 
                alpha=0.7, label=f'Trip {trip_id}')
    
    ax1.set_xlabel('Time of Day (Hours)', fontsize=12)
    ax1.set_ylabel('Trip ID', fontsize=12)
    ax1.set_title('Trip Timeline Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    
    # Plot 2: Data points per trip
    trip_nums = []
    point_counts = []
    
    for trip_idx, trip in enumerate(trips):
        trip_nums.append(trip_idx + 1)
        point_counts.append(len(trip))
    
    bars = ax2.bar(trip_nums, point_counts, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Trip ID', fontsize=12)
    ax2.set_ylabel('Number of Data Points', fontsize=12)
    ax2.set_title('Data Points per Trip', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, point_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_comprehensive_trip_analysis(trips):
    """
    Comprehensive visualization combining multiple perspectives of trip data.
    """
    if not trips:
        print("No trips to plot!")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Trip ID vs Data Point ID (scatter)
    trip_ids = []
    data_point_ids = []
    
    for trip_idx, trip in enumerate(trips):
        trip_id = trip_idx + 1
        for point_idx in range(len(trip)):
            trip_ids.append(trip_id)
            data_point_ids.append(point_idx + 1)
    
    ax1.scatter(data_point_ids, trip_ids, alpha=0.6, s=20, c='blue')
    ax1.set_xlabel('Data Point ID (within trip)')
    ax1.set_ylabel('Trip ID')
    ax1.set_title('Trip ID vs Data Point ID')
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks(range(1, len(trips) + 1))
    
    # Plot 2: Cumulative data points across trips
    cumulative_points = []
    trip_boundaries = [0]
    
    for trip in trips:
        if cumulative_points:
            cumulative_points.extend(range(cumulative_points[-1] + 1, cumulative_points[-1] + len(trip) + 1))
        else:
            cumulative_points.extend(range(1, len(trip) + 1))
        trip_boundaries.append(cumulative_points[-1])
    
    trip_ids_cumulative = []
    for trip_idx, trip in enumerate(trips):
        trip_ids_cumulative.extend([trip_idx + 1] * len(trip))
    
    ax2.plot(cumulative_points, trip_ids_cumulative, 'o-', markersize=3, alpha=0.7)
    for boundary in trip_boundaries[1:-1]:  # Skip first (0) and last
        ax2.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Cumulative Data Point ID')
    ax2.set_ylabel('Trip ID')
    ax2.set_title('Cumulative Data Points Across All Trips')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Data points per trip (bar chart)
    trip_nums = list(range(1, len(trips) + 1))
    point_counts = [len(trip) for trip in trips]
    
    bars = ax3.bar(trip_nums, point_counts, alpha=0.7, color='green')
    ax3.set_xlabel('Trip ID')
    ax3.set_ylabel('Number of Data Points')
    ax3.set_title('Data Points per Trip')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trip duration vs data points
    durations = []
    for trip in trips:
        start_time = trip['Data Actual Time'].iloc[0]
        end_time = trip['Data Actual Time'].iloc[-1]
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        durations.append(duration)
    
    ax4.scatter(point_counts, durations, alpha=0.7, s=50, c='purple')
    ax4.set_xlabel('Number of Data Points')
    ax4.set_ylabel('Trip Duration (minutes)')
    ax4.set_title('Trip Duration vs Data Points')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(point_counts, durations)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
# Example usage:
def main():
    """
    Main function to demonstrate trip extraction.
    Replace this with your actual data loading.
    """
    
    # Sample data structure (replace with your actual data loading)
    # df = pd.read_csv('your_gps_data.csv')
    
    df = pd.read_csv("data/finalOutput.csv")
    # Extract trips
    trips = extract_trips_from_gps_data(df, 
                                       proximity_threshold=0.1,  # 100m radius
                                       min_trip_duration_minutes=30,
                                       min_trip_points=100)
    
    # Analyze trips
    analyze_trips(trips)
    
    # Plot trips (if you have matplotlib installed)
    plot_trips(trips)
    
    return trips

# Advanced trip extraction with additional features
def enhanced_trip_extraction(df, proximity_threshold=0.1, min_trip_duration_minutes=30, 
                           min_trip_points=10, speed_threshold=5, idle_time_threshold=300):
    """
    Enhanced trip extraction that considers speed and idle times.
    
    Additional parameters:
    - speed_threshold: Minimum average speed (km/h) for a valid trip
    - idle_time_threshold: Maximum idle time (seconds) before considering trip ended
    """
    
    df_sorted = df.copy()
    df_sorted['Data Actual Time'] = pd.to_datetime(df_sorted['Data Actual Time'], format='%d-%m-%Y %I:%M:%S %p')
    df_sorted = df_sorted.sort_values('Data Actual Time').reset_index(drop=True)
    df_sorted['Date'] = df_sorted['Data Actual Time'].dt.date
    
    trips = []
    
    for date in df_sorted['Date'].unique():
        daily_data = df_sorted[df_sorted['Date'] == date].copy().reset_index(drop=True)
        
        if len(daily_data) < min_trip_points:
            continue
        
        i = 0
        while i < len(daily_data) - min_trip_points:
            trip_start_idx = i
            
            # Look for trip completion
            for j in range(i + min_trip_points, len(daily_data)):
                start_lat = daily_data.iloc[trip_start_idx]['Latitude']
                start_lon = daily_data.iloc[trip_start_idx]['Longitude']
                current_lat = daily_data.iloc[j]['Latitude']
                current_lon = daily_data.iloc[j]['Longitude']
                
                distance = geodesic((start_lat, start_lon), (current_lat, current_lon)).kilometers
                
                if distance <= proximity_threshold:
                    # Check trip validity
                    trip_data = daily_data.iloc[trip_start_idx:j+1].copy()
                    trip_duration = (trip_data['Data Actual Time'].iloc[-1] - 
                                   trip_data['Data Actual Time'].iloc[0]).total_seconds() / 60
                    
                    if trip_duration >= min_trip_duration_minutes:
                        # Calculate average speed
                        total_distance = 0
                        for k in range(1, len(trip_data)):
                            lat1, lon1 = trip_data.iloc[k-1]['Latitude'], trip_data.iloc[k-1]['Longitude']
                            lat2, lon2 = trip_data.iloc[k]['Latitude'], trip_data.iloc[k]['Longitude']
                            total_distance += geodesic((lat1, lon1), (lat2, lon2)).kilometers
                        
                        avg_speed = (total_distance / trip_duration) * 60  # km/h
                        
                        if avg_speed >= speed_threshold:
                            trip_data['Trip_ID'] = f"Trip_{date}_{len(trips)+1}"
                            trips.append(trip_data)
                            i = j  # Move to end of current trip
                            break
            else:
                i += 1
    
    return trips

