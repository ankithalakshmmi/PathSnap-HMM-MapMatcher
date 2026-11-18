"""Folium map generation (GPS, road network, reference route, predicted routes)."""
import folium
import numpy as np

from .data import Trip
from .pipeline import dedup

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
 padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
<b>{title}</b><br>{rows}</div>"""


def _arc_lines(trip, arcs):
    return [[(trip.node_lonlat[trip.arcs[a, 0], 1], trip.node_lonlat[trip.arcs[a, 0], 0]),
             (trip.node_lonlat[trip.arcs[a, 1], 1], trip.node_lonlat[trip.arcs[a, 1], 0])] for a in arcs]


def trip_map(trip: Trip, routes: dict = None, show_network=True, show_reference=True, title=None,
             max_gps=4000):
    """routes: {layer_name: (arc_sequence, colour)}"""
    lat, lon = trip.lonlat[:, 1], trip.lonlat[:, 0]
    m = folium.Map(tiles="OpenStreetMap")
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    legend = []

    if show_network:
        pad = 0.003
        inb = lambda ll: ((ll[:, 0] > lon.min() - pad) & (ll[:, 0] < lon.max() + pad) &
                          (ll[:, 1] > lat.min() - pad) & (ll[:, 1] < lat.max() + pad))
        keep = np.where(inb(trip.node_lonlat[trip.arcs[:, 0]]) | inb(trip.node_lonlat[trip.arcs[:, 1]]))[0]
        fg = folium.FeatureGroup("Road network", show=True)
        folium.PolyLine(_arc_lines(trip, keep), color="#9e9e9e", weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("#9e9e9e", "Road network"))

    if show_reference and trip.route is not None:
        fg = folium.FeatureGroup("Reference route", show=True)
        folium.PolyLine(_arc_lines(trip, dedup(trip.route)), color="#2ca02c", weight=7, opacity=0.5).add_to(fg)
        fg.add_to(m)
        legend.append(("#2ca02c", "Reference route"))

    for name, (arcs, colour) in (routes or {}).items():
        fg = folium.FeatureGroup(name, show=True)
        folium.PolyLine(_arc_lines(trip, dedup(arcs)), color=colour, weight=3.5, opacity=0.9).add_to(fg)
        fg.add_to(m)
        legend.append((colour, name))

    fg = folium.FeatureGroup("Raw GPS observations", show=True)
    step = max(1, len(lat) // max_gps)
    for la, lo, t in zip(lat[::step], lon[::step], trip.t[::step]):
        folium.CircleMarker((la, lo), radius=2, color="#1f77b4", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}s").add_to(fg)
    fg.add_to(m)
    legend.append(("#1f77b4", f"Raw GPS (every {step} pt)" if step > 1 else "Raw GPS"))

    folium.Marker((lat[0], lon[0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    rows = "".join(f'<span style="color:{c}">&#9632;</span> {n}<br>' for c, n in legend)
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=title or f"Trip {trip.trip_id}", rows=rows)))
    return m
