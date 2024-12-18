from shapely.strtree import STRtree
from pydantic import BaseModel
import logging
import time
import geopandas as gpd
from typing import Optional, List
from rtree import index

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class CalculationResult(BaseModel):
    geom1_area: float
    geom2_area: float
    overlap_area: float


class TopologyChecker:
    def __init__(self):
        self.spatial_index = None

    def get_utm_epsg(self, lat: float, lon: float) -> int:
        utm_band = str(int((lon + 180) / 6) + 1)
        if lat > 0:
            return int('326' + utm_band.zfill(2))
        else:
            return int('327' + utm_band.zfill(2))

    def calculate_areas(self, geom1, geom2, overlap_geom) -> CalculationResult:
        centroid = geom1.centroid
        utm_epsg = self.get_utm_epsg(centroid.y, centroid.x)

        geom1_utm = gpd.GeoSeries([geom1], crs='EPSG:4326').to_crs(f'EPSG:{utm_epsg}').iloc[0]
        geom2_utm = gpd.GeoSeries([geom2], crs='EPSG:4326').to_crs(f'EPSG:{utm_epsg}').iloc[0]
        overlap_utm = gpd.GeoSeries([overlap_geom], crs='EPSG:4326').to_crs(f'EPSG:{utm_epsg}').iloc[0]

        return CalculationResult(
            geom1_area=geom1_utm.area,
            geom2_area=geom2_utm.area,
            overlap_area=overlap_utm.area
        )

    def create_spatial_index(self, gdf: gpd.GeoDataFrame) -> index.Index:
        idx = index.Index()
        for i, geom in enumerate(gdf.geometry):
            idx.insert(i, geom.bounds)
        return idx

    def check_overlaps(self, gdf1: gpd.GeoDataFrame, gdf2: Optional[gpd.GeoDataFrame] = None) -> List[dict]:
        # Ensure correct CRS
        gdf1 = gdf1 if gdf1.crs == 'EPSG:4326' else gdf1.to_crs('EPSG:4326')
        if gdf2 is not None:
            gdf2 = gdf2 if gdf2.crs == 'EPSG:4326' else gdf2.to_crs('EPSG:4326')
        else:
            gdf2, self_check = gdf1, True

        geometries = gdf2.geometry.values
        polygon_overlaps = {}
        invalid_count = 0

        logger.info("Building STRtree spatial index...")
        tree = STRtree(geometries)

        start_time = time.time()
        processed = 0

        def process_overlap(idx1: int, idx2: int, geom1, geom2) -> None:
            overlap_geom = geom1.intersection(geom2)
            if overlap_geom.area <= 0:
                return

            areas = self.calculate_areas(geom1, geom2, overlap_geom)

            # Update or initialize first polygon data
            if idx1 not in polygon_overlaps:
                polygon_overlaps[idx1] = {
                    'geometry': geom1,
                    'total_overlap_area': areas.overlap_area,
                    'overlapping_with': {idx2},
                    'original_area': areas.geom1_area
                }
            else:
                polygon_overlaps[idx1]['total_overlap_area'] += areas.overlap_area
                polygon_overlaps[idx1]['overlapping_with'].add(idx2)

            # Update or initialize second polygon data
            if idx2 not in polygon_overlaps:
                polygon_overlaps[idx2] = {
                    'geometry': geom2,
                    'total_overlap_area': areas.overlap_area,
                    'overlapping_with': {idx1},
                    'original_area': areas.geom2_area
                }
            else:
                polygon_overlaps[idx2]['total_overlap_area'] += areas.overlap_area
                polygon_overlaps[idx2]['overlapping_with'].add(idx1)

        # Main processing loop
        for idx1, geom1 in enumerate(gdf1.geometry.values):
            if idx1 % 5000 == 0:
                elapsed = time.time() - start_time
                rate = idx1 / elapsed if elapsed > 0 else 0
                logger.info(f"Processing feature {idx1}/{len(geometries)} ({rate:.1f} features/sec)")

            if not geom1.is_valid:
                invalid_count += 1
                continue

            try:
                for idx2 in tree.query(geom1):
                    if self_check and idx2 <= idx1:
                        continue

                    processed += 1
                    if processed % 10000 == 0:
                        logger.info(f"Checked {processed} potential intersections")

                    geom2 = geometries[idx2]
                    if geom2.is_valid and geom1.intersects(geom2) and not geom1.touches(geom2):
                        try:
                            process_overlap(idx1, idx2, geom1, geom2)
                        except Exception as e:
                            invalid_count += 1
                            logger.debug(f"Error processing intersection: {e}")
            except Exception as e:
                invalid_count += 1
                logger.debug(f"Error processing feature {idx1}: {e}")

        if invalid_count:
            logger.warning(f"Skipped {invalid_count} invalid geometry pairs")

        return [{
            'error_type': 'major_overlap' if (data['total_overlap_area'] / data['original_area']) * 100 > 20 else 'minor_overlap',
            'feature_id': idx,
            'geometry': data['geometry'],
            'overlap_percentage': (data['total_overlap_area'] / data['original_area']) * 100,
            'total_overlap_area_m2': data['total_overlap_area'],
            'original_area_m2': data['original_area'],
            'overlapping_with': sorted(list(data['overlapping_with'])),
            'remarks': f'{"Major" if (data["total_overlap_area"] / data["original_area"]) * 100 > 20 else "Minor"} overlap '
                      f'({(data["total_overlap_area"] / data["original_area"]) * 100:.2f}%) with {len(data["overlapping_with"])} polygons'
        } for idx, data in polygon_overlaps.items() if data['original_area'] > 0]