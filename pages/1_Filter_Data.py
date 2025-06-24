import streamlit as st
import pandas as pd
from utils import VehicleStateAnalyzer
from visuals import extract_trips_from_gps_data

st.title("Filter and View Data")

if 'trips' not in st.session_state or 'data' not in st.session_state:
    st.warning("Please upload and process a CSV file on the main page first.")
    st.stop()

df = st.session_state['data']
trips = st.session_state['trips']

trip_ids = [trips[i]["Trip_ID"].iloc[0] for i in range(len(trips))]
selected_trip = st.selectbox("Select Trip_ID to filter", options=trip_ids)
for i in range(len(trip_ids)):
    if trip_ids[i] == selected_trip:
        filtered_df = trips[i]
        st.write(f"Showing data for Trip_ID: {selected_trip}")
        st.dataframe(filtered_df)
        break

# trip_ids = [trips[i]["Trip_ID"].iloc[0] for i in range(len(trips))]

# print(trip_ids)


