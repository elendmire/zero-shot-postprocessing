"""Figure 1: map of the German stations used in the study.

(a) Germany within central Europe with the station domain outlined; (b) the
stations coloured by altitude, with the two excluded stations marked.
Reads results/station_metadata.parquet (written by error_metrics.py), writes
figures/fig01_study_area.{pdf,svg}. Needs cartopy; the Natural Earth
boundaries are downloaded on first use.
"""
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import make_figures as mf

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    style = mf.STYLE
    meta = pd.read_parquet(REPO_ROOT / "results" / "station_metadata.parquet")
    used = meta[meta["used_in_study"]]
    excluded = meta[~meta["used_in_study"]]

    pad = 0.6
    lon0, lon1 = meta["longitude"].min() - pad, meta["longitude"].max() + pad
    lat0, lat1 = meta["latitude"].min() - pad, meta["latitude"].max() + pad

    mf._apply_rcparams()
    proj = ccrs.LambertConformal(central_longitude=10.0, central_latitude=51.0)
    fig = plt.figure(figsize=mf._figsize_in("double", 120))
    ax_loc = fig.add_axes([0.02, 0.30, 0.42, 0.62], projection=proj)
    ax_map = fig.add_axes([0.50, 0.12, 0.36, 0.80], projection=ccrs.PlateCarree())
    cax = fig.add_axes([0.89, 0.25, 0.015, 0.55])

    border = cfeature.NaturalEarthFeature("cultural", "admin_0_countries", "50m")
    land_color, sea_color, line_color = "#F2F2F2", "#FFFFFF", "#7F7F7F"

    ax_loc.set_extent([-5, 25, 42, 58], crs=ccrs.PlateCarree())
    ax_loc.add_feature(border, facecolor=land_color, edgecolor=line_color, linewidth=0.4)
    ax_loc.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=sea_color)
    for rec in cfeature.shapereader.Reader(
        cfeature.shapereader.natural_earth("50m", "cultural", "admin_0_countries")
    ).records():
        if rec.attributes.get("ADM0_A3") == "DEU":
            ax_loc.add_geometries([rec.geometry], ccrs.PlateCarree(), facecolor="#BFBFBF",
                                  edgecolor="#404040", linewidth=0.6)
    ax_loc.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0],
                transform=ccrs.PlateCarree(), color="black", linewidth=1.0)
    ax_loc.set_title("(a) Germany and the station domain", loc="left")

    ax_map.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
    ax_map.add_feature(border, facecolor=land_color, edgecolor=line_color, linewidth=0.6)
    ax_map.add_feature(cfeature.RIVERS.with_scale("50m"), edgecolor="#9EC9E2", linewidth=0.5)
    sc = ax_map.scatter(used["longitude"], used["latitude"], c=used["altitude_m"], cmap="viridis",
                        s=28, edgecolor="black", linewidth=0.4, transform=ccrs.PlateCarree(), zorder=3)
    ax_map.scatter(excluded["longitude"], excluded["latitude"], marker="x", color="black", s=30,
                   linewidth=1.0, transform=ccrs.PlateCarree(), zorder=3)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.4, color=mf.COLORS["gridline"],
                          x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": style["font"]["tick_label_pt"]}
    ax_map.set_title(f"(b) {len(used)} stations used in this study", loc="left")
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Station altitude (m)")
    ax_map.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markerfacecolor="#7F7F7F", markeredgecolor="black",
                   label=f"Used ({len(used)})"),
            Line2D([], [], marker="x", linestyle="", color="black",
                   label=f"Excluded, missing observations ({len(excluded)})"),
        ],
        loc="upper left", bbox_to_anchor=(0.02, 0.24), bbox_transform=fig.transFigure,
        frameon=False, fontsize=style["font"]["legend_pt"],
    )
    mf._save_figure(fig, "fig01_study_area", ["results/station_metadata.parquet"])
    plt.close(fig)


if __name__ == "__main__":
    main()
