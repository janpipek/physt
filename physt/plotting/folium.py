from __future__ import absolute_import

import folium

types = ("map")

dims = {
    "map" : [2],
}

def bins_to_json(h2):
    """

    :type h2: physt.histogram_nd.Histogram2D
    :param map:
    :return:
    """
    x0 = h2.get_bin_left_edges(0)
    x1 = h2.get_bin_right_edges(0)
    y0 = h2.get_bin_left_edges(1)
    y1 = h2.get_bin_right_edges(1)
    return {
        "type":"FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [x0[i], y0[j]],
                        [x1[i], y0[j]],
                        [x1[i], y1[j]],
                        [x0[i], y1[j]],
                        [x0[i], y0[j]]]],
                },
                "properties" : {
                    "count": float(h2.frequencies[i, j])
                }
            }
            for i in range(h2.shape[0])
            for j in range(h2.shape[1])
        ]
    }


def map(h2, map=None, tiles='stamentoner', cmap="wk", alpha=0.5, lw=1):
    """

    :type h2: physt.histogram_nd.Histogram2D
    :param map:
    :return:
    """
    if not map:
        posx = h2.get_bin_centers(0).mean()
        posy = h2.get_bin_centers(1).mean()
        zoom_start = 10
        map = folium.Map(location=[posy, posx], tiles=tiles)
    geo_json = bins_to_json(h2)

    from branca.colormap import LinearColormap

    color_map = LinearColormap(cmap, vmin=h2.frequencies.min(), vmax=h2.frequencies.max())

    #xx = h2.frequencies.max()

    # folium.

    def styling_function(bin):
        count = bin["properties"]["count"]
        return {
            "fillColor": color_map(count),
            "color": "black",
            "fillOpacity": alpha if count > 0 else 0,
            "strokeWidth": lw
        }# .update(styling)

    folium.GeoJson(geo_json, style_function=styling_function).add_to(map)
    return map