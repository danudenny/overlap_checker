import streamlit as st
import geopandas as gpd
import pandas as pd
import json
from topology_checker import TopologyChecker
import tempfile
import os
import numpy as np
from shapely.geometry import mapping

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
                    overlap_errors = checker.check_overlaps(gdf)
                    errors_df = gpd.GeoDataFrame(
                        overlap_errors,
                        crs='EPSG:4326'
                    )

                    # Display summary
                    st.success(f"Found {len(overlap_errors)} overlapping features")

                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["Summary", "Details", "Download"])

                    with tab1:
                        # Summary statistics
                        major_overlaps = len([e for e in overlap_errors if e['error_type'] == 'major_overlap'])
                        minor_overlaps = len([e for e in overlap_errors if e['error_type'] == 'minor_overlap'])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Major Overlaps (>20%)", major_overlaps)
                        with col2:
                            st.metric("Minor Overlaps (≤20%)", minor_overlaps)

                    with tab2:
                        # Detailed results
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

                    with tab3:
                        # Download options
                        col1, col2 = st.columns(2)

                        with col1:
                            # CSV download
                            if not errors_df.empty:
                                # Prepare CSV data
                                csv_df = errors_df.copy()
                                csv_df = csv_df.drop(columns=['geometry'])

                                # Convert numeric columns
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
                                # Convert to GeoJSON with custom conversion
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
            st.exception(e)  # This will show the full traceback


if __name__ == "__main__":
    main()