import numpy as np
import pandas as pd
from datetime import datetime

class VehicleStateAnalyzer:
    def __init__(self, idle_threshold=2, accel_threshold=2, time_window=2):
        """
        Initialize the analyzer with thresholds
        
        Parameters:
        - idle_threshold: Speed below which vehicle is considered idle (km/h)
        - accel_threshold: Speed change threshold for acceleration/retarding (km/h)
        - time_window: Number of records on each side to consider for context analysis
        """
        self.idle_threshold = idle_threshold
        self.accel_threshold = accel_threshold
        self.time_window = time_window
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two GPS points using Haversine formula (in meters)
        """
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371000  # Earth radius in meters
        return c * r
    
    def analyze_vehicle_states(self, df):
        """
        Analyze vehicle states based on speed patterns and GPS movement
        
        Expected DataFrame columns:
        - Speed: Vehicle speed in km/h
        - Latitude: GPS latitude
        - Longitude: GPS longitude
        - Any timestamp column (optional)
        """
        if df.empty:
            return df
        
        df = df.copy()
        df = df.reset_index(drop=True)
        
        # Ensure we have the required columns
        required_cols = ['Speed', 'Latitude', 'Longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate speed changes
        df['speed_change'] = df['Speed'].diff()
        
        # Calculate GPS distance moved between consecutive points
        df['distance_moved'] = 0.0
        for i in range(1, len(df)):
            df.loc[i, 'distance_moved'] = self.calculate_distance(
                df.loc[i-1, 'Latitude'], df.loc[i-1, 'Longitude'],
                df.loc[i, 'Latitude'], df.loc[i, 'Longitude']
            )
        
        # Calculate rolling statistics for context
        window_size = (self.time_window * 2) + 1  # Records on each side + current
        df['speed_rolling_mean'] = df['Speed'].rolling(window=window_size, center=True, min_periods=1).mean()
        df['speed_rolling_std'] = df['Speed'].rolling(window=window_size, center=True, min_periods=1).std()
        
        # Determine vehicle state for each record
        states = []
        confidence_scores = []
        
        for i in range(len(df)):
            state, confidence = self._determine_state_with_context(df, i)
            states.append(state)
            confidence_scores.append(confidence)
        
        df['vehicle_state'] = states
        df['confidence'] = confidence_scores
        
        return df
    
    def _determine_state_with_context(self, df, current_idx):
        """
        Determine vehicle state for a specific record using surrounding context
        """
        current_speed = df.loc[current_idx, 'Speed']
        current_distance = df.loc[current_idx, 'distance_moved']
        speed_change = df.loc[current_idx, 'speed_change'] if pd.notna(df.loc[current_idx, 'speed_change']) else 0
        
        # Get context window
        start_idx = max(0, current_idx - self.time_window)
        end_idx = min(len(df), current_idx + self.time_window + 1)
        
        context_speeds = df.loc[start_idx:end_idx, 'Speed'].tolist()
        context_distances = df.loc[start_idx:end_idx, 'distance_moved'].tolist()
        
        # Rule 1: IDLE - Low speed and minimal movement
        if current_speed <= self.idle_threshold:
            if current_distance < 5:  # Less than 5 meters moved
                return 'idle', 0.95
            elif current_speed == 0:
                return 'idle', 0.90
            else:
                return 'idle', 0.70  # Low speed but some movement
        
        # Rule 2: Check speed trend from context for ACCELERATING/RETARDING
        if len(context_speeds) >= 3:
            # Calculate trend from context speeds
            speed_changes = []
            for j in range(1, len(context_speeds)):
                speed_changes.append(context_speeds[j] - context_speeds[j-1])
            
            avg_speed_change = np.mean(speed_changes) if speed_changes else 0
            
            # Strong acceleration pattern
            if avg_speed_change >= self.accel_threshold:
                confidence = min(0.90, 0.60 + (avg_speed_change / 10))
                return 'accelerating', confidence
            
            # Strong retarding pattern
            elif avg_speed_change <= -self.accel_threshold:
                confidence = min(0.90, 0.60 + (abs(avg_speed_change) / 10))
                return 'retarding', confidence
        
        # Rule 3: Single point acceleration/retarding
        if abs(speed_change) >= self.accel_threshold:
            if speed_change > 0:
                confidence = min(0.80, 0.50 + (speed_change / 15))
                return 'accelerating', confidence
            else:
                confidence = min(0.80, 0.50 + (abs(speed_change) / 15))
                return 'retarding', confidence
        
        # Rule 4: CRUISING - Stable speed above idle threshold
        if current_speed > self.idle_threshold:
            # Check speed stability in context
            if len(context_speeds) >= 3:
                speed_std = np.std(context_speeds)
                if speed_std <= 3:  # Low variation indicates stable cruising
                    confidence = max(0.70, 0.90 - (speed_std / 10))
                    return 'cruising', confidence
            
            # Default cruising for speeds above idle
            return 'cruising', 0.60
        
        # Fallback
        return 'idle' if current_speed <= self.idle_threshold else 'cruising', 0.40
    
    def generate_summary_stats(self, df):
        """
        Generate summary statistics for the analyzed data
        """
        if df.empty or 'vehicle_state' not in df.columns:
            return {}
        
        total_records = len(df)
        state_counts = df['vehicle_state'].value_counts()
        
        # Calculate distance by state
        distance_by_state = df.groupby('vehicle_state')['distance_moved'].sum()
        
        stats = {
            'total_records': total_records,
            'total_distance_meters': df['distance_moved'].sum(),
            'state_distribution': {
                'counts': state_counts.to_dict(),
                'percentages': (state_counts / total_records * 100).round(2).to_dict(),
                'distance_meters': distance_by_state.to_dict()
            },
            'speed_stats': {
                'average_speed': df['Speed'].mean(),
                'max_speed': df['Speed'].max(),
                'min_speed': df['Speed'].min(),
                'speed_std': df['Speed'].std()
            },
            'average_confidence': df['confidence'].mean()
        }
        
        return stats