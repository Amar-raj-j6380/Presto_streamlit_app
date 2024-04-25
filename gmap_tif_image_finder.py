import streamlit as st
import gmaps
import geopandas as gpd
from shapely.geometry import Polygon
from tempfile import NamedTemporaryFile

import rasterio
from rasterio.features import geometry_window
import rasterio.features
import rasterio.warp

import folium
import numpy as np
import matplotlib.pyplot as plt


def get_lat_lon(tif_path):
    """
    Get the latitude and longitude of a given Tiff image.

    Args:
        tif_path: The path to the Tiff image.

    Returns:
        A tuple of (latitude, longitude).
    """

    with rasterio.open(tif_path) as dataset:
        try:
            # Read the dataset's valid data mask as a ndarray.
            mask = dataset.dataset_mask()

            # Extract feature shapes and values from the array.
            for geom, val in rasterio.features.shapes(
                    mask, transform=dataset.transform):

                # Transform shapes from the dataset's own coordinate
                # reference system to CRS84 (EPSG:4326).
                geom = rasterio.warp.transform_geom(
                    dataset.crs, 'EPSG:4326', geom, precision=6)

                # Print GeoJSON shapes to stdout.
            print("i am in try::",geom)
            return(geom)
        except Exception as inst:
            print("exception")
            return(inst)

def main():

    st.title("TIF Image Location Finder")

    st.markdown("""Upload a TIF image to find its location on Google Maps.""")

    uploaded_file = st.file_uploader("Upload TIFF file", type=["tif", "tiff"])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded file
        with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as src:
            st.write(f"Number of bands: {src.count}")
            st.write(f"Width: {src.width}")
            st.write(f"Height: {src.height}")
            st.title("Polygon Plotter on Google Maps")
            tif_path = 'C:/Users/dell/Desktop/vam.tif'
            print(type(get_lat_lon(tmp_path))==dict)
            if(type(get_lat_lon(tmp_path))==dict):
                cord_list = get_lat_lon(tmp_path)['coordinates'][0]
                print(cord_list)
                m = folium.Map(location=cord_list[0], zoom_start=12)

                trail_coordinates = cord_list

                folium.PolyLine(trail_coordinates, tooltip="Coast").add_to(m)
                folium_map = m._repr_html_()
                st.components.v1.html(folium_map, width=700, height=500)
            else:
                st.title("Image is corrupted")

if __name__ == "__main__":
    main()