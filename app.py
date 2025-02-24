import streamlit as st
import geopandas as gpd
import pandas as pd
import json
from topology_checker import TopologyChecker
import tempfile
import os
import numpy as np
from shapely.geometry import mapping, Polygon, MultiPolygon, LineString, Point, GeometryCollection
from shapely import wkt
import folium
from streamlit_folium import folium_static
from loguru import logger

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
        if row['major_overlap']:
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
            }
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

def ensure_polygon_new(geom):
    try:
        if isinstance(geom, Polygon):
            return geom
        elif isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda p: p.area)
        elif isinstance(geom, (LineString, Point)):
            return geom.buffer(1e-8)
        elif isinstance(geom, GeometryCollection):
            polygons = [g for g in geom.geoms if isinstance(g, Polygon)]
            if polygons:
                return max(polygons, key=lambda p: p.area)
            else:
                largest_geom = max(
                    geom.geoms, key=lambda g: g.area if hasattr(g, "area") else 0
                )
                return ensure_polygon_new(largest_geom)
        else:
            return Polygon(geom)
    except Exception as e:
        logger.error(f"Error in ensure_polygon_new: {e}")
        return Polygon()

def fix_minor_overlaps(gdf, overlap_pairs):
    try:
        gdf["geometry"] = gdf["geometry"].apply(wkt.loads)

        for unique_id1, unique_id2, _ in overlap_pairs:
            poly1 = gdf.loc[gdf["uid"] == unique_id1, "geometry"].values[0]
            poly2 = gdf.loc[gdf["uid"] == unique_id2, "geometry"].values[0]

            intersection = poly1.intersection(poly2)

            if not intersection.is_empty:
                new_poly1 = poly1.difference(poly2).buffer(-0.00001)
                new_poly2 = poly2

                new_poly1 = ensure_polygon_new(new_poly1)
                new_poly2 = ensure_polygon_new(new_poly2)

                gdf.loc[gdf["uid"] == unique_id1, "geometry"] = new_poly1
                gdf.loc[gdf["uid"] == unique_id2, "geometry"] = new_poly2

        gdf["polygon_corrected"] = gdf["geometry"].apply(
            lambda geom: geom.wkt if geom else None
        )
        gdf.drop(columns=["geometry"], inplace=True)
        return gdf

    except Exception as e:
        logger.error(f"Error in fix_overlapping_geometries: {e}")
        raise
        

def create_fixed_overlap_map(original_gdf, fixed_gdf):
    """Create a Folium map comparing original and fixed features"""
    # Get the center of all geometries
    center_lat = original_gdf.geometry.centroid.y.mean()
    center_lon = original_gdf.geometry.centroid.x.mean()

    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add original features
    folium.GeoJson(
        original_gdf,
        name='Original Features',
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 1,
            'fillOpacity': 0.3
        }
    ).add_to(m)

    # Add fixed features
    folium.GeoJson(
        fixed_gdf,
        name='Fixed Features',
        style_function=lambda x: {
            'fillColor': 'green',
            'color': 'green',
            'weight': 1,
            'fillOpacity': 0.3
        }
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Fit bounds to show all features
    try:
        bounds = original_gdf.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception as e:
        st.warning(f"Could not fit map bounds: {str(e)}")

    return m


def main():
    st.title("Topology Overlap Checker")

    # Initialize session state variables
    if 'fixed_gdf' not in st.session_state:
        st.session_state.fixed_gdf = None
    if 'original_gdf' not in st.session_state:
        st.session_state.original_gdf = None
    if 'overlap_errors' not in st.session_state:
        st.session_state.overlap_errors = None
    if 'errors_df' not in st.session_state:
        st.session_state.errors_df = None
    if 'check_clicked' not in st.session_state:
        st.session_state.check_clicked = False

    uploaded_file = st.file_uploader("Choose a GeoJSON file", type=['geojson'])

    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_file = save_uploadedfile(uploaded_file)

            # Load the data
            with st.spinner('Loading data...'):
                gdf = gpd.read_file(temp_file)
                st.success(f"Loaded {len(gdf)} features")
                st.session_state.original_gdf = gdf

            # Initialize checker
            checker = TopologyChecker()

            # Process button
            check_button = st.button("Check Overlaps")
            if check_button:
                st.session_state.check_clicked = True

            if st.session_state.check_clicked:
                with st.spinner('Processing overlaps...'):
                    if st.session_state.overlap_errors is None:
                        logger.info(gdf.columns)
                        gdf.set_geometry("geometry")
                        gdf.set_crs("EPSG:4326")
                        st.session_state.overlap_errors = checker.check_overlaps(gdf)

                        errors_df = gpd.GeoDataFrame(
                            st.session_state.overlap_errors,
                        )
                        
                        # Explicitly set the geometry column
                        errors_df.set_geometry("polygon", inplace=True)
                        
                        # Set the CRS
                        errors_df.set_crs("EPSG:4326", inplace=True)
                        if 'polygon' in errors_df.columns:
                            errors_df.rename(columns={"polygon": "geometry"}, inplace=True)
                        
                        if 'geometry' in errors_df.columns:
                            errors_df.set_geometry("geometry", inplace=True)
                        
                        st.session_state.errors_df = errors_df

                    # Display summary
                    st.success(f"Found {len(st.session_state.overlap_errors)} overlapping features")

                    # Display results in tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Original Map", "Fixed Map", "Compare Maps", "Download"])

                    with tab1:
                        # Summary statistics
                        major_overlaps = len([e for e in st.session_state.overlap_errors if e['major_overlap']])
                        minor_overlaps = len([e for e in st.session_state.overlap_errors if not e['major_overlap']])

                        # Create 3 columns for metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Features", len(st.session_state.original_gdf))
                        with col2:
                            st.metric("Major Overlaps (>20%)", major_overlaps)
                        with col3:
                            st.metric("Minor Overlaps (≤20%)", minor_overlaps)

                        # Add Fix Minor Overlaps button
                        if minor_overlaps > 0:
                            fix_button = st.button("Fix Minor Overlaps")
                            if fix_button:
                                try:
                                    # Prepare overlap pairs for minor overlaps
                                    minor_overlap_pairs = [
                                        (e['uid'], e['overlapping_with'], e['overlap_percentage'])
                                        for e in st.session_state.overlap_errors
                                        if not e['major_overlap']
                                    ]
                                    
                                    # Make sure polygon_corrected column exists
                                    if 'polygon_corrected' not in st.session_state.original_gdf.columns:
                                        st.session_state.original_gdf['polygon_corrected'] = st.session_state.original_gdf['geometry'].apply(lambda x: x.wkt)
                                    
                                    # Fix minor overlaps
                                    fixed_gdf = fix_minor_overlaps(st.session_state.original_gdf.copy(), minor_overlap_pairs)
                                    st.session_state.fixed_gdf = fixed_gdf
                                    
                                    st.success("Minor overlaps have been fixed! Check the Fixed Map tab to view the results.")
                                    
                                except Exception as e:
                                    st.error(f"Error fixing overlaps: {str(e)}")
                                    logger.error(f"Error in fix_minor_overlaps: {e}")

                        # Display detailed table
                        st.subheader("Detailed Results")
                        if not st.session_state.errors_df.empty:
                            display_cols = [
                                'major_overlap', 'minor_overlap', 'uid', 'overlap_percentage',
                                'total_overlap_area_m2', 'original_area_m2',
                                'overlapping_with', 'remarks'
                            ]
                            display_df = st.session_state.errors_df[display_cols].copy()

                            # Convert numeric columns to float for display
                            numeric_cols = ['overlap_percentage', 'total_overlap_area_m2', 'original_area_m2']
                            for col in numeric_cols:
                                display_df[col] = display_df[col].astype(float)

                            st.dataframe(display_df)

                    with tab2:
                        # Original Map visualization
                        st.subheader("Original Features with Overlaps")
                        overlap_map = create_overlap_map(st.session_state.original_gdf, st.session_state.errors_df)
                        folium_static(overlap_map)

                    with tab3:
                        # Fixed Map visualization
                        st.subheader("Fixed Features")
                        if st.session_state.fixed_gdf is not None:
                            fixed_map = create_fixed_overlap_map(st.session_state.original_gdf, st.session_state.fixed_gdf)
                            folium_static(fixed_map)
                        else:
                            st.info("No fixed features available yet. Use the 'Fix Minor Overlaps' button in the Summary tab to generate fixed features.")

                    with tab4:
                        # Side by side comparison
                        st.subheader("Compare Original vs Fixed Features")
                        if st.session_state.fixed_gdf is not None:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Original Features")
                                overlap_map = create_overlap_map(st.session_state.original_gdf, st.session_state.errors_df)
                                folium_static(overlap_map)
                            with col2:
                                st.write("Fixed Features")
                                fixed_map = create_fixed_overlap_map(st.session_state.original_gdf, st.session_state.fixed_gdf)
                                folium_static(fixed_map)
                        else:
                            st.info("No fixed features available yet. Use the 'Fix Minor Overlaps' button in the Summary tab to generate fixed features.")

                    with tab5:
                        # Download options
                        st.subheader("Download Results")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # Original errors CSV download
                            if not st.session_state.errors_df.empty:
                                csv_df = st.session_state.errors_df.copy()
                                csv_df = csv_df.drop(columns=['geometry'])
                                csv = csv_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Original Errors CSV",
                                    data=csv,
                                    file_name="overlap_errors.csv",
                                    mime="text/csv"
                                )

                        with col2:
                            # Original errors GeoJSON download
                            if not st.session_state.errors_df.empty:
                                geojson_str = convert_to_geojson(st.session_state.errors_df)
                                st.download_button(
                                    label="Download Original Errors GeoJSON",
                                    data=geojson_str,
                                    file_name="overlap_errors.geojson",
                                    mime="application/json"
                                )

                        with col3:
                            # Fixed features GeoJSON download
                            if st.session_state.fixed_gdf is not None:
                                fixed_geojson = convert_to_geojson(st.session_state.fixed_gdf)
                                st.download_button(
                                    label="Download Fixed Features GeoJSON",
                                    data=fixed_geojson,
                                    file_name="fixed_geometries.geojson",
                                    mime="application/json"
                                )

            # Cleanup
            os.unlink(temp_file)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
