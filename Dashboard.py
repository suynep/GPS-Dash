import streamlit as st
import pandas as pd
from utils import VehicleStateAnalyzer
from visuals import extract_trips_from_gps_data

st.title("Upload CSV File")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df  # Save in session_state for multipage sharing
        st.success("File uploaded successfully!")
        st.write(df.head())

        def preprocess_data(df):
            unneeded_columns = ['Odometer', 'Other Ports', 'Data Received Time', 'IGN']
            week_filtered_df = df.drop(columns=unneeded_columns) 
            # making sure that each record's `Data String` attrib contains the `Satellites` param
            week_filtered_df = week_filtered_df[week_filtered_df["Data String"].str.contains('Satellites')] 
            # create a new column called "Satellites" containing the integral value from `Data String`
            week_filtered_df['Satellites'] = week_filtered_df['Data String'].str.extract(r'\[Satellites=(\d+)\]', expand=False).astype(int)
            MIN_SAT_NUM = 10 # future-proofing za work :)
            week_satellite_adjusted_df = week_filtered_df[week_filtered_df["Satellites"] >= MIN_SAT_NUM] # final data w/ proper satellite >= 10 values

            df = week_satellite_adjusted_df.copy() # too difficult to type the name, thus we shorten it :)
            df = df[df["Data String"].str.contains('Angle')] 
            df['Angle'] = df['Data String'].str.extract(r'\[Angle=(\d+)\]', expand=False).astype(int)
            df = df[df["Data String"].str.contains('Altitude')] 
            df['Altitude'] = df['Data String'].str.extract(r'\[Altitude=(\d+)\]', expand=False).astype(int)
            # As a final task in this step, we format the lat and lon values into standard Decimal Degree Format
            df['Latitude'] = df['Latitude'] / 1e7
            df['Longitude'] = df['Longitude'] / 1e7

            return df

        df = preprocess_data(df)
        analyzer = VehicleStateAnalyzer(idle_threshold=2, accel_threshold=2, time_window=2)
        df = analyzer.analyze_vehicle_states(df)
        stats = analyzer.generate_summary_stats(df)
        st.session_state['data'] = df  # Re-Save in session_state for multipage sharing

        trips = extract_trips_from_gps_data(df, 
                           proximity_threshold=0.1,  # 100m radius
                           min_trip_duration_minutes=30,
                           min_trip_points=100)


        st.session_state['trips'] = trips
                            
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
else:
    st.info("Please upload a CSV file to continue.")

