import streamlit as st
import pandas as pd
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import great_circle

st.set_page_config(layout="wide")
st.title("📦 Customer & Hub Visualization Tool")
# Footer note
st.markdown("<div style='text-align:right; font-size:12px; color:gray;'>Version 1.0.2 Developed by Jidapa Buranachan</div>", unsafe_allow_html=True)

# Downloadable template section
st.markdown("### 📥 Download Template Files")
cust_template = pd.DataFrame(columns=["Customer_Code", "Lat", "Long", "Type", "Province"])
dc_template = pd.DataFrame(columns=["Hub_Name", "Lat", "Long", "Type", "Province"])

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="⬇️ Download Customer Template",
        data=cust_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='Customer_Template.csv',
        mime='text/csv'
    )
with col2:
    st.download_button(
        label="⬇️ Download Hub Template",
        data=dc_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='Hub_Template.csv',
        mime='text/csv'
    )

# Upload files
cust_file = st.file_uploader("Upload Customer File (.csv with Lat, Long, Customer_Code, Type, Province)", type="csv")
dc_file = st.file_uploader("Upload Hub File (.csv with Lat, Long, Hub_Name, Type, Province)", type="csv")

if cust_file:
    try:
        cust_data = pd.read_csv(cust_file)
        if cust_data.empty:
            st.error("🚫 Customer CSV file is empty. Please upload a file with data.")
            st.stop()
        cust_data = cust_data.dropna(subset=['Lat', 'Long'])
    except pd.errors.EmptyDataError:
        st.error("🚫 Customer CSV file is empty or corrupted.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load customer file: {e}")
        st.stop()

    locations = cust_data[['Lat', 'Long']]

    # Filter by Type
    customer_types = cust_data['Type'].dropna().unique().tolist()
    selected_types = st.multiselect("Filter Customer Types:", options=customer_types, default=customer_types)
    cust_data = cust_data[cust_data['Type'].isin(selected_types)]

    st.subheader("📍 Nearest Hub for Each Customer")

    if dc_file:
        dc_data = pd.read_csv(dc_file).dropna(subset=['Lat', 'Long'])

        # Filter Hub types
        dc_types = dc_data['Type'].dropna().unique().tolist()
        selected_dc_types = st.multiselect("Filter Hub Types:", options=dc_types, default=dc_types)
        dc_data = dc_data[dc_data['Type'].isin(selected_dc_types)]

        # Find nearest hub for each customer
        def find_nearest_dc(cust_lat, cust_lon, dc_df):
            distances = dc_df.apply(
                lambda row: great_circle((cust_lat, cust_lon), (row['Lat'], row['Long'])).km,
                axis=1
            )
            idx = distances.idxmin()
            return dc_df.loc[idx, 'Hub_Name'], distances.min()

        results = []
        for _, row in cust_data.iterrows():
            nearest_dc, min_dist = find_nearest_dc(row['Lat'], row['Long'], dc_data)
            results.append({
                'Customer_Code': row['Customer_Code'],
                'Type': row.get('Type', 'Unknown'),
                'Province': row.get('Province', 'N/A'),
                'Nearest_Hub': nearest_dc,
                'Distance_km': round(min_dist, 2)
            })

        nearest_df = pd.DataFrame(results)
        st.dataframe(nearest_df)

    # Suggested hub clustering (KMeans)
    st.subheader("🌐 Suggested Hub Locations (KMeans)")
    n_dc = st.slider("Select number of suggested hubs (KMeans):", 1, 10, 5)
    kmeans = KMeans(n_clusters=n_dc, random_state=42)
    kmeans.fit(cust_data[['Lat', 'Long']])
    kmeans_dc_locations = kmeans.cluster_centers_

    # Radius-based Hub Suggestion (separate logic)
    st.subheader("🧭 Radius-based Hub Grouping (DBSCAN-like)")
    radius_km = st.slider("Radius per hub (km):", 10, 300, 100, key="radius_custom")
    radius_m = radius_km * 1000

    from geopy.distance import geodesic
    import numpy as np

    custom_dc_locations = []
    unassigned = cust_data.copy()
    while not unassigned.empty:
        center = unassigned[['Lat', 'Long']].iloc[0].values
        in_radius = unassigned.apply(
            lambda row: geodesic(center, (row['Lat'], row['Long'])).km <= radius_km,
            axis=1
        )
        group = unassigned[in_radius]
        custom_dc_locations.append(group[['Lat', 'Long']].mean().values)
        unassigned = unassigned[~in_radius]

    # Choose one for map rendering (KMeans or custom)
    clustering_choice = st.radio("Select clustering result to display on map:", ["KMeans", "Radius-based"])
    dc_locations = kmeans_dc_locations if clustering_choice == "KMeans" else custom_dc_locations

    # Layer visibility controls
    show_heatmap = st.checkbox("Show Heatmap", value=True)
    show_province_circles = st.checkbox("Show Customer Province Circles", value=True)
    show_customer_markers = st.checkbox("Show Customer Markers", value=True)
    show_existing_hubs = st.checkbox("Show Existing Hubs", value=True)
    show_suggested_hubs = st.checkbox("Show Suggested Hubs", value=True)
    
    # Create folium map
    m = folium.Map(location=[13.75, 100.5], zoom_start=6)

    # Layer: Heatmap
    heatmap_layer = FeatureGroup(name="Heatmap")
    if show_heatmap:
        HeatMap(cust_data[['Lat', 'Long']].values.tolist(), radius=10).add_to(heatmap_layer)
    if show_heatmap:
        heatmap_layer.add_to(m)

    # Province-based Circle Visualization
    province_layer = FeatureGroup(name="Customer Circles")
    province_counts = cust_data['Province'].value_counts()
    for province, count in province_counts.items():
        subset = cust_data[cust_data['Province'] == province]
        if not subset.empty:
            lat, lon = subset[['Lat', 'Long']].mean()
            # Dynamic popup listing first few customer codes
            sample_customers = subset['Customer_Code'].head(5).tolist()
            sample_text = '<br>'.join(sample_customers)
            full_popup = f"<b>{province}</b>: {count} customers<br><hr><b>Sample:</b><br>{sample_text}"
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + count**0.5,
                color='blue',
                fill=True,
                fill_opacity=0.5,
                popup=folium.Popup(full_popup, max_width=250)
            ).add_to(province_layer)
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + count**0.5,
                color='blue',
                fill=True,
                fill_opacity=0.5,
                popup=f"{province}: {count} customers"
            ).add_to(province_layer)
    if show_province_circles:
        province_layer.add_to(m)

    # Layer: Customer markers
    customer_layer = FeatureGroup(name="Customer Markers")
    customer_cluster = MarkerCluster(name="Customer Cluster")
    for _, row in cust_data.iterrows():
        type_lower = row.get('Type', '').lower()
        icon = 'home'
        color = 'lightblue' if type_lower == 'lotus' else 'red'
        popup_text = f"Customer: {row['Customer_Code']} ({row.get('Type', 'Unknown')})<br>Province: {row.get('Province', 'N/A')}"
        folium.Marker(
            [row['Lat'], row['Long']],
            popup=popup_text,
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(customer_cluster)
    customer_cluster.add_to(customer_layer)
    if show_customer_markers:
        customer_layer.add_to(m)

    # Existing hubs by Type
    hub_layer = FeatureGroup(name="Existing Hubs")
    if dc_file:
        for _, row in dc_data.iterrows():
            type_lower = row.get('Type', '').lower()
            icon = 'store'
            color = 'lightblue' if type_lower == 'lotus' else 'red'
            popup_text = f"Hub: {row['Hub_Name']} ({row.get('Type', 'Unknown')})<br>Province: {row.get('Province', 'N/A')}"
            folium.Marker(
                [row['Lat'], row['Long']],
                popup=popup_text,
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(hub_layer)
    if show_existing_hubs:
        hub_layer.add_to(m)

    # Suggested hubs
    suggested_layer = FeatureGroup(name="Suggested Hubs")
    for i, (lat, lon) in enumerate(dc_locations):
        folium.Marker(
            location=[lat, lon],
            popup=f"Suggest Hub #{i+1}",
            icon=folium.Icon(color='green', icon='star', prefix='fa')
        ).add_to(suggested_layer)

        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color='green',
            fill=True,
            fill_opacity=0.1,
            popup=f"Radius {radius_km} km"
        ).add_to(suggested_layer)
    if show_suggested_hubs:
        suggested_layer.add_to(m)

    # Layer toggle
    LayerControl().add_to(m)

    # Show final map
    st.subheader("🗺️ Visualization")
    st_folium(m, width=1100, height=600, returned_objects=[], key="main_map")

