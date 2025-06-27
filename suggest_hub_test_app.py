import streamlit as st
import pandas as pd
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import great_circle, geodesic
import numpy as np

import geopandas as gpd
from shapely.geometry import Point

#------------------------------------------------------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Customer & Hub Visualization Tool")
# Footer note
st.markdown("<div style='text-align:right; font-size:12px; color:gray;'>Version 1.0.3 Developed by Jidapa Buranachan</div>", unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------------------


# Downloadable template section
st.markdown("### Download Template Files")
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

#------------------------------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------------------------------
    # โหลดแผนที่ประเทศไทย
    thailand = gpd.read_file("thailand.geojson")
    thailand_union = thailand_gdf.unary_union
    
    # แปลงลูกค้าเป็น GeoDataFrame
    cust_data['geometry'] = cust_data.apply(lambda row: Point(row['Long'], row['Lat']), axis=1)
    cust_gdf = gpd.GeoDataFrame(cust_data, geometry='geometry', crs="EPSG:4326")
    
    # กรองเฉพาะลูกค้าที่อยู่ในขอบเขตประเทศไทยจริง
    cust_gdf = cust_gdf[cust_gdf.geometry.within(thailand.unary_union)]
    
    # คืนค่า DataFrame กลับไปใช้งานต่อ
    cust_data = pd.DataFrame(cust_gdf.drop(columns='geometry'))

    #Filter: only customers within Thailand bounding box
    #THAI_BOUNDING_BOX = {
    #    'min_lat': 5.61,
    #    'max_lat': 20.46,
    #    'min_lon': 97.34,
    #    'max_lon': 105.64
    #}
    #cust_data = cust_data[
    #    (cust_data['Lat'] >= THAI_BOUNDING_BOX['min_lat']) &
    #    (cust_data['Lat'] <= THAI_BOUNDING_BOX['max_lat']) &
    #    (cust_data['Long'] >= THAI_BOUNDING_BOX['min_lon']) &
    #    (cust_data['Long'] <= THAI_BOUNDING_BOX['max_lon'])
    #]


#------------------------------------------------------------------------------------------------------------------------
    
    # Filter by Type
    customer_types = cust_data['Type'].dropna().unique().tolist()
    selected_types = st.multiselect("Filter Customer Types:", options=customer_types, default=customer_types)
    cust_data = cust_data[cust_data['Type'].isin(selected_types)]

#------------------------------------------------------------------------------------------------------------------------
    
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

  
    #------------------------------------------------------------------------------------------------------------------------
        
        # Suggest New Hubs for Out-of-Radius Customers
        st.subheader("Suggest New Hubs Based on Radius & Existing Hubs")
        radius_threshold_km = st.slider("Set Radius Threshold from Existing Hubs (km):", 10, 500, 100)
        
    
    #------------------------------------------------------------------------------------------------------------------------
        
        def is_outside_hubs(lat, lon):
            return all(geodesic((lat, lon), (hub_lat, hub_lon)).km > radius_threshold_km for hub_lat, hub_lon in dc_data[['Lat', 'Long']].values)
        
        cust_data['Outside_Hub'] = cust_data.apply(lambda row: is_outside_hubs(row['Lat'], row['Long']), axis=1)
        outside_customers = cust_data[cust_data['Outside_Hub'] == True]
        
        # ✅ เพิ่มตรวจสอบว่าอยู่ในประเทศไทยจริงด้วย polygon
        outside_customers['geometry'] = outside_customers.apply(lambda row: Point(row['Long'], row['Lat']), axis=1)
        outside_gdf = gpd.GeoDataFrame(outside_customers, geometry='geometry', crs="EPSG:4326")
        outside_gdf = outside_gdf[outside_gdf.geometry.within(thailand_union)]
        
        st.markdown(f"<b>{len(outside_gdf)} customers</b> are outside the {radius_threshold_km} km range from existing hubs AND within Thailand.", unsafe_allow_html=True)
        
        if not outside_gdf.empty:
            n_new_hubs = st.slider("How many new hubs to suggest for uncovered areas?", 1, 10, 3)
            new_hub_kmeans = KMeans(n_clusters=n_new_hubs, random_state=42)
            new_hub_kmeans.fit(outside_gdf[['Lat', 'Long']])
        
            # ✅ ตรวจสอบศูนย์กลางว่าอยู่ในประเทศไทย
            new_hub_locations = []
            for lat, lon in new_hub_kmeans.cluster_centers_:
                if Point(lon, lat).within(thailand_union):
                    new_hub_locations.append((lat, lon))
                else:
                    print(f"⚠️ Hub suggestion at ({lat:.3f}, {lon:.3f}) is outside Thailand and was skipped.")

            st.subheader("New Hub Suggestions Map")
            m_new = folium.Map(location=[13.75, 100.5], zoom_start=6, control_scale=True)
            
     #------------------------------------------------------------------------------------------------------------------------
                
        # Layer visibility controls
            with st.expander("🧭 Layer Visibility Controls"):
                show_heatmap = st.checkbox("Show Heatmap", value=True)
                show_customer_markers = st.checkbox("Show Customer Markers", value=True)
                show_existing_hubs = st.checkbox("Show Existing Hubs", value=True)
                show_suggested_hubs = st.checkbox("Show Suggested Hubs", value=True)
                show_hub_radius_layer = st.checkbox("Show Existing Hub Radius Zones", value=True)
                           
      
    #------------------------------------------------------------------------------------------------------------------------
    
            # Existing hub layer
            existing_layer = FeatureGroup(name="Existing Hubs")
            for _, row in dc_data.iterrows():
                folium.Marker(
                    location=[row['Lat'], row['Long']],
                    popup=row['Hub_Name'],
                    icon=folium.Icon(color = 'red' if row.get('Type', '').lower() == 'makro' else 'blue', icon='store', prefix='fa')
                ).add_to(existing_layer)
            if show_existing_hubs:
                existing_layer.add_to(m_new)

            # Existing hub radius circles
            if show_hub_radius_layer:
                radius_layer = FeatureGroup(name="Existing Hub Radius")
                for _, row in dc_data.iterrows():
                    folium.Circle(
                        location=[row['Lat'], row['Long']],
                        radius=radius_threshold_km * 1000,
                        color='gray',
                        fill=False,
                        dash_array="5"
                    ).add_to(radius_layer)
                radius_layer.add_to(m_new)

            # Outside customer layer with brand-based color
            outside_layer = FeatureGroup(name="Outside Customers")
            for _, row in outside_customers.iterrows():
                color = 'red' if row.get('Type', '').lower() == 'makro' else 'blue'
                folium.CircleMarker(
                    location=[row['Lat'], row['Long']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.5,
                    popup=row['Customer_Code']
                ).add_to(outside_layer)
            if show_customer_markers:
                outside_layer.add_to(m_new)

            # Suggested hub layer
            suggest_layer = FeatureGroup(name="Suggested New Hubs")
            for i, (lat, lon) in enumerate(new_hub_locations):
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Suggest New Hub #{i+1}",
                    icon=folium.Icon(color='darkgreen', icon='star', prefix='fa')
                ).add_to(suggest_layer)
                folium.Circle(
                    location=[lat, lon],
                    radius=radius_threshold_km * 1000,
                    color='darkgreen',
                    fill=True,
                    fill_opacity=0.1,
                    popup=f"Radius {radius_threshold_km} km"
                ).add_to(suggest_layer)
            if show_suggested_hubs:
                suggest_layer.add_to(m_new)

            # Combined heatmap
            if show_heatmap:
                heatmap_layer = FeatureGroup(name="Customer Heatmap")
                HeatMap(
                    cust_data[['Lat', 'Long']].values.tolist(),
                    radius=10,
                    gradient={0.2: '#FFE5B4', 0.6: '#FFA500', 1: '#FF8C00'}
                ).add_to(heatmap_layer)
                heatmap_layer.add_to(m_new)

            LayerControl().add_to(m_new)

        #------------------------------------------------------------------------------------------------------------------------
            
            st_folium(m_new, width=1100, height=600, key="new_hub_map", returned_objects=[], feature_group_to_add=None, center=[13.75, 100.5], zoom=6)
