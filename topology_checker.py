from functools import lru_cache
from typing import Dict, List, Optional, Set

import geopandas as gpd
import numpy as np
from loguru import logger
from pydantic import BaseModel
from shapely.geometry import Polygon
from shapely.strtree import STRtree


class CalculationResult(BaseModel):
    geom1_area: float
    geom2_area: float
    overlap_area: float


class TopologyChecker:
    def __init__(self):
        self.spatial_index = None
        self._utm_cache = {}
        self._transformed_geoms = {}

    @lru_cache(maxsize=1024)
    def get_utm_epsg(self, lat: float, lon: float) -> int:
        """Cached UTM EPSG code calculation"""
        utm_band = str(int((lon + 180) / 6) + 1)
        return int(f"{'326' if lat > 0 else '327'}{utm_band.zfill(2)}")

    def _transform_to_utm(self, geom: Polygon, utm_epsg: int) -> Polygon:
        """Transform geometry to UTM with caching"""
        geom_hash = hash(geom.wkb)
        cache_key = (geom_hash, utm_epsg)

        if cache_key not in self._transformed_geoms:
            transformed = (
                gpd.GeoSeries([geom], crs="EPSG:4326")
                .to_crs(f"EPSG:{utm_epsg}")
                .iloc[0]
            )
            self._transformed_geoms[cache_key] = transformed

        return self._transformed_geoms[cache_key]

    def calculate_areas(
        self, geom1: Polygon, geom2: Polygon, overlap_geom: Polygon
    ) -> CalculationResult:
        """Calculate areas using cached UTM projections"""
        centroid = geom1.centroid
        utm_epsg = self.get_utm_epsg(centroid.y, centroid.x)

        geom1_utm = self._transform_to_utm(geom1, utm_epsg)
        geom2_utm = self._transform_to_utm(geom2, utm_epsg)
        overlap_utm = self._transform_to_utm(overlap_geom, utm_epsg)

        return CalculationResult(
            geom1_area=geom1_utm.area,
            geom2_area=geom2_utm.area,
            overlap_area=overlap_utm.area,
        )

    def _determine_overlap_type(self, polygon_overlaps: Dict[int, Dict]) -> Dict[int, bool]:
        """
        Determine final overlap types considering connected polygons.
        Returns a dictionary mapping polygon index to whether it has a major overlap (True) or not (False).
        """
        major_overlaps: Set[int] = set()
        overlap_types: Dict[int, bool] = {}
        
        # First pass: identify initial major overlaps
        for idx, data in polygon_overlaps.items():
            overlap_percentage = (data["total_overlap_area"] / data["original_area"]) * 100
            if overlap_percentage > 20:
                major_overlaps.add(idx)
        
        # Second pass: propagate major overlap status to connected minor overlaps
        changed = True
        while changed:
            changed = False
            for idx, data in polygon_overlaps.items():
                # Skip if already marked as major
                if idx in major_overlaps:
                    continue
                
                # Check if any connected polygon has a major overlap
                for connected_idx in data["overlapping_with"]:
                    if connected_idx in major_overlaps:
                        major_overlaps.add(idx)
                        changed = True
                        break
        
        # Create final mapping
        for idx in polygon_overlaps:
            overlap_types[idx] = idx in major_overlaps
            
        return overlap_types

    def check_overlaps(
        self, gdf1: gpd.GeoDataFrame, gdf2: Optional[gpd.GeoDataFrame] = None
    ) -> List[dict]:
        # Ensure correct CRS
        if not isinstance(gdf1, gpd.GeoDataFrame):
            gdf1["geometry"] = gdf1["polygon_original"] or gdf1["geometry"] or gdf1["polygon"]
            gdf1 = gpd.GeoDataFrame(gdf1)
            gdf1.set_geometry("geometry", inplace=True)
            gdf1 = gdf1.set_crs("EPSG:4326")

        gdf1 = gdf1 if gdf1.crs == "EPSG:4326" else gdf1.to_crs("EPSG:4326")
        if gdf2 is not None:
            gdf2 = gdf2 if gdf2.crs == "EPSG:4326" else gdf2.to_crs("EPSG:4326")
        else:
            gdf2, self_check = gdf1, True

        geometries = np.array(gdf2.geometry.values)
        polygon_overlaps: Dict[int, Dict] = {}
        invalid_count = 0

        logger.info("Building STRtree spatial index...")
        tree = STRtree(geometries)

        valid_geoms1 = np.array([geom.is_valid for geom in gdf1.geometry])
        valid_geoms2 = np.array([geom.is_valid for geom in geometries])

        batch_size = 1000

        for start_idx in range(0, len(gdf1), batch_size):
            end_idx = min(start_idx + batch_size, len(gdf1))
            batch_geoms = gdf1.geometry.values[start_idx:end_idx]

            for idx1, geom1 in enumerate(batch_geoms, start=start_idx):
                if not valid_geoms1[idx1]:
                    invalid_count += 1
                    continue

                try:
                    nearby_idxs = tree.query(geom1)
                    valid_nearby = [
                        idx2 for idx2 in nearby_idxs if valid_geoms2[idx2]
                    ]

                    for idx2 in valid_nearby:
                        if self_check and idx2 <= idx1:
                            continue

                        geom2 = geometries[idx2]

                        if geom1.intersects(geom2) and not geom1.touches(geom2):
                            try:
                                overlap_geom = geom1.intersection(geom2)
                                if overlap_geom.area <= 0:
                                    continue

                                areas = self.calculate_areas(
                                    geom1, geom2, overlap_geom
                                )

                                if idx1 not in polygon_overlaps:
                                    polygon_overlaps[idx1] = {
                                        "geometry": geom1,
                                        "total_overlap_area": areas.overlap_area,
                                        "overlapping_with": {idx2},
                                        "original_area": areas.geom1_area,
                                    }
                                else:
                                    polygon_overlaps[idx1][
                                        "total_overlap_area"
                                    ] += areas.overlap_area
                                    polygon_overlaps[idx1][
                                        "overlapping_with"
                                    ].add(idx2)

                                if idx2 not in polygon_overlaps:
                                    polygon_overlaps[idx2] = {
                                        "geometry": geom2,
                                        "total_overlap_area": areas.overlap_area,
                                        "overlapping_with": {idx1},
                                        "original_area": areas.geom2_area,
                                    }
                                else:
                                    polygon_overlaps[idx2][
                                        "total_overlap_area"
                                    ] += areas.overlap_area
                                    polygon_overlaps[idx2][
                                        "overlapping_with"
                                    ].add(idx1)

                            except Exception as e:
                                invalid_count += 1
                                logger.debug(
                                    f"Error processing intersection: {e}"
                                )

                except Exception as e:
                    invalid_count += 1
                    logger.debug(f"Error processing feature {idx1}: {e}")

            if len(self._transformed_geoms) > 10000:
                self._transformed_geoms.clear()

        if invalid_count:
            logger.warning(f"Skipped {invalid_count} invalid geometry pairs")

        # Determine final overlap types
        overlap_types = self._determine_overlap_type(polygon_overlaps)

        results = [
            {
                "major_overlap": overlap_types[idx],
                "minor_overlap": not overlap_types[idx],
                "uid": idx,
                "polygon": data["geometry"],
                "overlap_percentage": (
                    data["total_overlap_area"] / data["original_area"]
                ) * 100,
                "total_overlap_area_m2": data["total_overlap_area"],
                "original_area_m2": data["original_area"],
                "overlapping_with": sorted(list(data["overlapping_with"])),
                "remarks": (
                    f"{'Major' if overlap_types[idx] else 'Minor'} "
                    f"overlap ({(data['total_overlap_area'] / data['original_area']) * 100:.2f}%) "
                    f"with {len(data['overlapping_with'])} polygons"
                ),
            }
            for idx, data in polygon_overlaps.items()
            if data["original_area"] > 0
        ]

        return results
