"""PathSnap evaluator UI (Laptop 2).

Run:  streamlit run ui/app.py
Set the API address in the sidebar (e.g. http://192.168.1.23:8000), or via PATHSNAP_API.
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: GPS map-matching & route reconstruction")

api = st.sidebar.text_input("API URL (Laptop 1)", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")

try:
    resp = requests.get(f"{api}/trips", timeout=10)
    resp.raise_for_status()
    info = resp.json()
except Exception as e:
    st.error(f"Cannot reach API at {api}: {e}")
    st.stop()

st.sidebar.success(f"Connected · partition: {info['partition']}")
trip_id = st.selectbox(f"Select one of the {len(info['trip_ids'])} evaluation trips", info["trip_ids"])

if st.button("Run map-matching", type="primary"):
    with st.spinner(f"API is processing trip {trip_id} ..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    st.subheader(f"Trip {res['trip_id']} · {res['n_gps_points']} GPS points · "
                 f"processed in {res['timing_s']['total']} s")
    m = res.get("metrics", {})
    if m:
        h, c = m["classifier_hmm"], m["classifier_only"]
        cols = st.columns(5)
        cols[0].metric("Top-1 accuracy (HMM)", f"{h['top1_accuracy']:.2%}",
                       f"{h['top1_accuracy'] - c['top1_accuracy']:+.2%} vs classifier")
        cols[1].metric("Route F1", f"{h['route_f1']:.2%}")
        cols[2].metric("Route continuity", f"{h['continuity']:.2%}",
                       f"{h['continuity'] - c['continuity']:+.2%} vs classifier")
        cols[3].metric("Sequence LCS", f"{h['sequence_lcs']:.2%}")
        cols[4].metric("Points scored", h["n_points_scored"])
        with st.expander("All metrics"):
            st.json(m)

    gps = res["gps"]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap = folium.Map(tiles="OpenStreetMap")
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    fg = folium.FeatureGroup("Raw GPS")
    for p in gps:
        folium.CircleMarker(p, radius=2, color="#1f77b4", fill=True, weight=0, fill_opacity=.8).add_to(fg)
    fg.add_to(fmap)
    fg = folium.FeatureGroup("Classifier only", show=False)
    folium.PolyLine(res["classifier_route"], color="#ff7f0e", weight=3).add_to(fg)
    fg.add_to(fmap)
    fg = folium.FeatureGroup("Predicted route (HMM)")
    folium.PolyLine(res["predicted_route"], color="#d62728", weight=4).add_to(fg)
    fg.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st.caption("Blue dots: raw GPS · Red: predicted route (classifier + HMM/Viterbi) · "
               "Orange (toggle): classifier-only")
    st_folium(fmap, height=600, use_container_width=True, returned_objects=[])
