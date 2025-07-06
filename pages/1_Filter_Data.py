import streamlit as st
import pandas as pd
from utils import VehicleStateAnalyzer
from visuals import extract_trips_from_gps_data
import plotly.graph_objects as go
import plotly.express as px

st.title("Filter and View Data")

if "trips" not in st.session_state or "data" not in st.session_state:
    st.warning("Please upload and process a CSV file on the main page first.")
    st.stop()

df = st.session_state["data"]
trips = st.session_state["trips"]

trip_ids = [trips[i]["Trip_ID"].iloc[0] for i in range(len(trips))]
selected_trip = st.selectbox("Select Trip_ID to filter", options=trip_ids)


def plot_metric(label, value, suffix=""):
    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=value,
            number={"suffix": suffix, "font": {"size": 28}},
            title={"text": label, "font": {"size": 20}},
        )
    )
    fig.update_layout(height=120, margin=dict(t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)


for i in range(len(trip_ids)):
    if trip_ids[i] == selected_trip:
        filtered_df = trips[i]
        # --- Metrics Frame ---
        top_left_column, top_right_column = st.columns(2)
        with top_left_column:
            col1, col2 = st.columns(2)
            with col1:
                plot_metric("Avg. Speed", filtered_df["Speed"].mean(), " km/h")
                plot_metric("Avg. Altitude", filtered_df["Altitude"].mean(), " m")
            with col2:
                plot_metric("Avg. Angle", filtered_df["Angle"].mean(), "°")
                plot_metric("Avg. Satellites", int(filtered_df["Satellites"].mean()), "")

        # --- Pie Chart: Full Width ---
        st.markdown("### Vehicle State Distribution")
        vehicle_state_counts = filtered_df['vehicle_state'].value_counts().reset_index()
        vehicle_state_counts.columns = ['vehicle_state', 'count']
        fig_pie = px.pie(
            vehicle_state_counts,
            names='vehicle_state',
            values='count',
            title=None,
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.write(f"Showing data for Trip_ID: {selected_trip}")
        st.dataframe(filtered_df)
        break


# trip_ids = [trips[i]["Trip_ID"].iloc[0] for i in range(len(trips))]

# print(trip_ids)
