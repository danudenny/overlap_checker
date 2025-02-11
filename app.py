import streamlit as st
import geopandas as gpd
import pandas as pd
import json
from topology_checker import TopologyChecker
import tempfile
import os
import numpy as np
from shapely.geometry import mapping
import folium
from streamlit_folium import folium_static
import plotly.express as px
from loguru import logger
import plotly.graph_objects as go

st.set_page_config(page_title="Topology Overlap Checker", layout="wide")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_to_geojson(gdf):
    """Convert GeoDataFrame to GeoJSON with proper type conversion"""
    feature_collection = {
        "type": "FeatureCollection",
        "features": []
    }

    for idx, row in gdf.iterrows():
        feature = {
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": {}
        }

        # Convert all properties to native Python types
        for column in gdf.columns:
            if column != 'geometry':
                value = row[column]
                if isinstance(value, (np.integer, np.int64)):
                    value = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    value = float(value)
                elif isinstance(value, list) and value and isinstance(value[0], (np.integer, np.int64)):
                    value = [int(v) for v in value]
                feature["properties"][column] = value

        feature_collection["features"].append(feature)

    return json.dumps(feature_collection, cls=NumpyEncoder)


def save_uploadedfile(uploadedfile):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_file:
        tmp_file.write(uploadedfile.getvalue())
        return tmp_file.name


def create_overlap_map(gdf, errors_df):
    """Create a Folium map with original and overlapping features"""
    # Get the center of all geometries
    center_lat = errors_df.geometry.centroid.y.mean()
    center_lon = errors_df.geometry.centroid.x.mean()

    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add original features with simpler style function
    folium.GeoJson(
        gdf,
        name='Original Features',
        style_function=lambda x: {
            'fillColor': 'gray',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.1
        }
    ).add_to(m)

    # Create feature groups for different overlap types
    major_group = folium.FeatureGroup(name="Major Overlaps")
    minor_group = folium.FeatureGroup(name="Minor Overlaps")

    # Add overlapping features
    for _, row in errors_df.iterrows():
        if row['error_type'] == 'major_overlap':
            color = 'red'
            feature_group = major_group
        else:
            color = 'yellow'
            feature_group = minor_group

        folium.GeoJson(
            mapping(row['geometry']),  # Convert geometry to GeoJSON
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': c,
                'weight': 2,
                'fillOpacity': 0.5
            },
            tooltip=f"ID: {row['feature_id']}<br>Overlap: {row['overlap_percentage']:.2f}%"
        ).add_to(feature_group)

    # Add feature groups to map
    major_group.add_to(m)
    minor_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Fit bounds to show all features
    try:
        bounds = gdf.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception as e:
        st.warning(f"Could not fit map bounds: {str(e)}")

    return m


def create_statistics_plots(errors_df):
    """Create statistical visualizations using plotly"""
    # Overlap type distribution
    fig_types = px.pie(
        errors_df,
        names='error_type',
        title='Distribution of Overlap Types',
        color='error_type',
        color_discrete_map={'major_overlap': 'red', 'minor_overlap': 'yellow'}
    )

    # Overlap percentage distribution
    fig_dist = px.histogram(
        errors_df,
        x='overlap_percentage',
        title='Distribution of Overlap Percentages',
        nbins=20,
        color='error_type'
    )

    # Top 10 largest overlaps
    top_overlaps = errors_df.nlargest(10, 'overlap_percentage')
    fig_top = px.bar(
        top_overlaps,
        x='feature_id',
        y='overlap_percentage',
        title='Top 10 Largest Overlaps',
        color='error_type'
    )

    return fig_types, fig_dist, fig_top


def main():
    st.title("Topology Overlap Checker")

    uploaded_file = st.file_uploader("Choose a GeoJSON file", type=['geojson'])

    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_file = save_uploadedfile(uploaded_file)

            # Load the data
            with st.spinner('Loading data...'):
                gdf = gpd.read_file(temp_file)
                st.success(f"Loaded {len(gdf)} features")

            # Initialize checker
            checker = TopologyChecker()

            # Process button
            if st.button("Check Overlaps"):
                with st.spinner('Processing overlaps...'):
                    logger.info(gdf.columns)
                    gdf.set_geometry("geometry")
                    gdf.set_crs("EPSG:4326")
                    overlap_errors = checker.check_overlaps(gdf)
                    errors_df = gpd.GeoDataFrame(
                        overlap_errors,
                        geometry='geometry'
                        crs='EPSG:4326'
                    )

                    # Display summary
                    st.success(f"Found {len(overlap_errors)} overlapping features")

                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Map", "Statistics", "Download"])

                    with tab1:
                        # Summary statistics
                        major_overlaps = len([e for e in overlap_errors if e['error_type'] == 'major_overlap'])
                        minor_overlaps = len([e for e in overlap_errors if e['error_type'] == 'minor_overlap'])

                        # Create 3 columns for metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Features", len(gdf))
                        with col2:
                            st.metric("Major Overlaps (>20%)", major_overlaps)
                        with col3:
                            st.metric("Minor Overlaps (≤20%)", minor_overlaps)

                        # Display detailed table
                        st.subheader("Detailed Results")
                        if not errors_df.empty:
                            display_cols = [
                                'error_type', 'feature_id', 'overlap_percentage',
                                'total_overlap_area_m2', 'original_area_m2',
                                'overlapping_with', 'remarks'
                            ]
                            display_df = errors_df[display_cols].copy()

                            # Convert numeric columns to float for display
                            numeric_cols = ['overlap_percentage', 'total_overlap_area_m2', 'original_area_m2']
                            for col in numeric_cols:
                                display_df[col] = display_df[col].astype(float)

                            st.dataframe(display_df)

                    with tab2:
                        # Map visualization
                        st.subheader("Overlap Map")
                        overlap_map = create_overlap_map(gdf, errors_df)
                        folium_static(overlap_map)

                    with tab3:
                        # Statistical visualizations
                        st.subheader("Statistical Analysis")

                        fig_types, fig_dist, fig_top = create_statistics_plots(errors_df)

                        # Display plots in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_types, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_dist, use_container_width=True)

                        # Display top overlaps chart full width
                        st.plotly_chart(fig_top, use_container_width=True)

                        # Additional statistics
                        st.subheader("Summary Statistics")
                        stats_df = pd.DataFrame({
                            'Metric': [
                                'Average Overlap Percentage',
                                'Maximum Overlap Percentage',
                                'Minimum Overlap Percentage',
                                'Total Overlap Area (m²)',
                                'Average Overlap Area (m²)'
                            ],
                            'Value': [
                                f"{errors_df['overlap_percentage'].mean():.2f}%",
                                f"{errors_df['overlap_percentage'].max():.2f}%",
                                f"{errors_df['overlap_percentage'].min():.2f}%",
                                f"{errors_df['total_overlap_area_m2'].sum():,.2f}",
                                f"{errors_df['total_overlap_area_m2'].mean():,.2f}"
                            ]
                        })
                        st.table(stats_df)

                    with tab4:
                        # Download options
                        st.subheader("Download Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            # CSV download
                            if not errors_df.empty:
                                csv_df = errors_df.copy()
                                csv_df = csv_df.drop(columns=['geometry'])

                                numeric_cols = ['overlap_percentage', 'total_overlap_area_m2', 'original_area_m2']
                                for col in numeric_cols:
                                    csv_df[col] = csv_df[col].astype(float)

                                csv = csv_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name="overlap_errors.csv",
                                    mime="text/csv"
                                )

                        with col2:
                            # GeoJSON download
                            if not errors_df.empty:
                                geojson_str = convert_to_geojson(errors_df)
                                st.download_button(
                                    label="Download GeoJSON",
                                    data=geojson_str,
                                    file_name="overlap_errors.geojson",
                                    mime="application/json"
                                )

            # Cleanup
            os.unlink(temp_file)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
