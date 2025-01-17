import folium
from folium.plugins import MarkerCluster
from branca.colormap import linear



class PlotOnMap:
    def __init__(self, data, target, save_path):
        self.data = data
        self.target = target
        self.save_path = save_path

    def plot_listings(self):
        # Create a map centered around a specific point
        map_center = [52.379189, 4.899431]  # Example center (Amsterdam)
        # Create a color scale (linear color map)
        colormap = linear.YlOrRd_09.scale(self.data[self.target].min(), self.data[self.target].max())
        m = folium.Map(location=map_center, zoom_start=5)
        # Create a marker cluster
        marker_cluster = MarkerCluster().add_to(m)

        # Add markers for each location
        for _, location in self.data.iterrows():
                folium.CircleMarker(
            location=[location['latitude'], location['longitude']],
            radius=10,  # size of the marker
            color=colormap(location[self.target]),  # color based on the target
            fill=True,
            fill_color=colormap(location[self.target]),
            fill_opacity=0.7,
            popup=f"target: {location[self.target]}"
            ).add_to(marker_cluster)

        # Save the map to an HTML file
        m.save(self.save_path)