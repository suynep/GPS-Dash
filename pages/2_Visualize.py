import streamlit as st
import pydeck as pdk
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


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


def plot_trip_from_index(trips, index=0):
    """
    Plot the extracted trip at the given index using matplotlib.
    """
    import matplotlib.pyplot as plt

    if not trips or index >= len(trips):
        st.warning("No trips to plot or index out of range!")
        return

    trip = trips[index]
    trip_id = trip['Trip_ID'].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the trip line
    ax.plot(trip['Longitude'], trip['Latitude'], 'o-', label=trip_id, alpha=0.7, markersize=3)

    # Start and end points
    ax.plot(trip['Longitude'].iloc[0], trip['Latitude'].iloc[0], 's', markersize=8, color='green', label='Start')
    ax.plot(trip['Longitude'].iloc[-1], trip['Latitude'].iloc[-1], '^', markersize=8, color='red', label='End')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Extracted Bus Trip in KTM Valley')
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)



st.title("Visualize Trip Data")


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
        st.write(f"Map data for Trip_ID: {selected_trip}")
        
        top_left_column, top_right_column = st.columns(2)
        with top_left_column:
            col1, col2 = st.columns(2)
            with col1:
                plot_metric("Avg. Speed", filtered_df["Speed"].mean(), " km/h")
                plot_metric("Avg. Altitude", filtered_df["Altitude"].mean(), " m")
            with col2:
                plot_metric("Avg. Angle", filtered_df["Angle"].mean(), "°")
                plot_metric("Avg. Satellites", int(filtered_df["Satellites"].mean()), "")
        # 3D plotting Begins

        st.markdown("### Vehicle State Legend")

        legend_items = {
            "Idle": "#C8C8C8",           # [200, 200, 200]
            "Accelerating": "#00FF00",   # [0, 255, 0]
            "Retarding": "#FF0000",      # [255, 0, 0]
            "Cruising": "#0000FF",       # [0, 0, 255]
        }

        cols = st.columns(len(legend_items))
        for col, (state, color) in zip(cols, legend_items.items()):
            col.markdown(
                f"<div style='display: flex; align-items: center;'>"
                f"<div style='width: 20px; height: 20px; background-color: {color}; "
                f"margin-right: 10px; border-radius: 4px;'></div>"
                f"<span style='font-size: 14px;'>{state}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        state_color_map = {
            'idle': [200, 200, 200],
            'accelerating': [0, 255, 0],
            'retarding': [255, 0, 0],
            'cruising': [0, 0, 255]
        }

        trip_df = filtered_df.copy()
        trip_df['vehicle_state'] = trip_df['vehicle_state'].fillna('idle').str.lower()
        trip_df['color'] = trip_df['vehicle_state'].map(state_color_map).apply(lambda x: [int(i) for i in x])
        trip_df['elevation'] = trip_df['Speed'].fillna(0).astype(float)
        trip_df['Longitude'] = trip_df['Longitude'].astype(float)
        trip_df['Latitude'] = trip_df['Latitude'].astype(float)

        data = trip_df[['Longitude', 'Latitude', 'elevation', 'color', 'Speed', 'vehicle_state']].to_dict(orient='records')

        column_layer = pdk.Layer(
            "ColumnLayer",
            data=data,
            get_position='[Longitude, Latitude]',
            get_elevation='elevation',
            elevation_scale=10,
            radius=6,
            get_fill_color='color',
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        view_state = pdk.ViewState(
            latitude=trip_df['Latitude'].mean(),
            longitude=trip_df['Longitude'].mean(),
            zoom=14,
            pitch=50,
        )

        deck = pdk.Deck(
            layers=[column_layer],
            initial_view_state=view_state,
            tooltip={"text": "Speed: {Speed} km/h\nState: {vehicle_state}"},
        )

        st.pydeck_chart(deck)
        

        # 3D plotting ENDS

        plot_trip_from_index(trips, index = i)        

        st.write(f"Frame data for Trip_ID: {selected_trip}")
        st.dataframe(filtered_df)


        break
