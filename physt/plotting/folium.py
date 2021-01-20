"""
Plotting inside maps with folium.

Very experimental!
"""
from typing import Any, Dict, Optional

import folium

from physt.histogram_nd import Histogram2D

types = ("geo_map",)

dims = {
    "geo_map": [2],
}


def _bins_to_json(h2: Histogram2D) -> Dict[str, Any]:
    """Create GeoJSON representation of histogram bins

    Parameters
    ----------
    h2: physt.histogram_nd.Histogram2D
        A histogram of coordinates (in degrees)

    Returns
    -------
    geo_json : dict
    """
    south = h2.get_bin_left_edges(0)
    north = h2.get_bin_right_edges(0)
    west = h2.get_bin_left_edges(1)
    east = h2.get_bin_right_edges(1)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    # Note that folium and GeoJson have them swapped
                    "coordinates": [
                        [
                            [east[j], south[i]],
                            [east[j], north[i]],
                            [west[j], north[i]],
                            [west[j], south[i]],
                            [east[j], south[i]],
                        ]
                    ],
                },
                "properties": {"count": float(h2.frequencies[i, j])},
            }
            for i in range(h2.shape[0])
            for j in range(h2.shape[1])
        ],
    }


def geo_map(
    h2: Histogram2D,
    *,
    map: Optional[folium.folium.Map] = None,
    tiles: str = "stamenterrain",
    cmap="wk",
    alpha: float = 0.5,
    lw=1,
    fit_bounds=None,
    layer_name=None,
) -> folium.folium.Map:
    """Show rectangular grid over a map.

    Parameters
    ----------
    h2: A histogram of coordinates (in degrees: latitude, longitude)
    map : The map to insert the histogram into

    Returns
    -------
    map : folium.folium.Map
    """
    if not map:
        latitude = h2.get_bin_centers(0).mean()
        longitude = h2.get_bin_centers(1).mean()
        map = folium.Map(location=[latitude, longitude], tiles=tiles)
        if fit_bounds is None:
            fit_bounds = True

    geo_json = _bins_to_json(h2)

    if not layer_name:
        layer_name = h2.name

    from branca.colormap import LinearColormap

    color_map = LinearColormap(cmap, vmin=h2.frequencies.min(), vmax=h2.frequencies.max())

    # legend = folium.Html("<div>Legend</div>")
    # legend_div = folium.Div("20%", "20%", "75%", "5%")
    #
    # legend_div.add_to(map)
    # legend_div.add_child(legend)

    # xx = h2.frequencies.max()

    def styling_function(bin):
        count = bin["properties"]["count"]
        return {
            "fillColor": color_map(count),
            "color": "black",
            "fillOpacity": alpha if count > 0 else 0,
            "weight": lw,
            # "strokeWidth": lw,
            "opacity": alpha if count > 0 else 0,
        }  # .update(styling)

    layer = folium.GeoJson(geo_json, style_function=styling_function, name=layer_name)
    layer.add_to(map)
    if fit_bounds:
        map.fit_bounds(layer.get_bounds())
    return map
