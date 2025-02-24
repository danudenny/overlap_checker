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

    # Create base map
    m = folium.Map(location=[0, 0], zoom_start=10)

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

def fix_minor_overlaps(gdf, minor_overlap_pairs):
    """Fix minor overlaps in the GeoDataFrame"""
    try:
        # Create a copy to avoid modifying the original
        working_gdf = gdf.copy()
        
        # Ensure geometry column exists and is properly set
        if 'geometry' in working_gdf.columns:
            working_gdf['geometry'] = working_gdf['geometry'].apply(
                lambda x: wkt.loads(x) if isinstance(x, str) else x
            )
        
        # Set the geometry column
        working_gdf = gpd.GeoDataFrame(working_gdf, geometry='geometry', crs="EPSG:4326")

        # Process each overlap pair
        for unique_id1, unique_id2, _ in minor_overlap_pairs:
            # Get the geometries
            mask1 = working_gdf['uid'] == unique_id1
            mask2 = working_gdf['uid'] == unique_id2
            
            if not any(mask1) or not any(mask2):
                continue
                
            poly1 = working_gdf.loc[mask1, 'geometry'].iloc[0]
            poly2 = working_gdf.loc[mask2, 'geometry'].iloc[0]

            if not (isinstance(poly1, (Polygon, MultiPolygon)) and isinstance(poly2, (Polygon, MultiPolygon))):
                continue

            # Fix any invalid geometries
            poly1 = poly1.buffer(0)
            poly2 = poly2.buffer(0)

            intersection = poly1.intersection(poly2)

            if not intersection.is_empty:
                new_poly1 = poly1.difference(poly2).buffer(-0.00001)
                new_poly2 = poly2

                new_poly1 = ensure_polygon_new(new_poly1)
                new_poly2 = ensure_polygon_new(new_poly2)

                working_gdf.loc[mask1, 'geometry'] = new_poly1
                working_gdf.loc[mask2, 'geometry'] = new_poly2

        # Update polygon_corrected column
        working_gdf['geometry'] = working_gdf['geometry'].apply(lambda x: x.wkt)
        
        return working_gdf

    except Exception as e:
        logger.error(f"Error in fix_minor_overlaps: {str(e)}")
        raise
        

def create_fixed_overlap_map(original_gdf, fixed_gdf):
    # Create base map
    m = folium.Map(location=[0, 0], zoom_start=10)

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
    if 'selected_identifier' not in st.session_state:
        st.session_state.selected_identifier = None

    uploaded_file = st.file_uploader("Choose a GeoJSON file", type=['geojson'])

    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_file = save_uploadedfile(uploaded_file)

            # Load the data
            with st.spinner('Loading data...'):
                gdf = gpd.read_file(temp_file)
                st.success(f"Loaded {len(gdf)} features")
                
                # Get all columns except geometry for selection
                available_columns = [col for col in gdf.columns if col != 'geometry']
                
                # Display column selector
                selected_column = st.selectbox(
                    "Select unique identifier column from properties",
                    options=available_columns,
                    help="Choose the column that contains unique identifiers for each feature"
                )

                # Verify column uniqueness
                if selected_column:
                    if gdf[selected_column].is_unique:
                        st.success(f"✅ Column '{selected_column}' contains unique values")
                        st.session_state.selected_identifier = selected_column
                        st.session_state.original_gdf = gdf
                    else:
                        st.error(f"❌ Column '{selected_column}' contains duplicate values. Please select a column with unique values.")
                        st.stop()

            # Initialize checker
            checker = TopologyChecker()

            # Process button
            check_button = st.button("Check Overlaps")
            if check_button and st.session_state.selected_identifier:
                st.session_state.check_clicked = True

            if st.session_state.check_clicked:
                with st.spinner('Processing overlaps...'):
                    if st.session_state.overlap_errors is None:
                        logger.info(gdf.columns)
                        gdf.set_geometry("geometry")
                        gdf.set_crs("EPSG:4326")
                        
                        # Use the selected identifier column instead of 'uid'
                        gdf['uid'] = gdf[st.session_state.selected_identifier]
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

                    # [Rest of the code remains the same...]

            # Cleanup
            os.unlink(temp_file)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
