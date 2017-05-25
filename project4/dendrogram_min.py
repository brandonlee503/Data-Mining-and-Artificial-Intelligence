import plotly.plotly as py
from plotly.graph_objs import *

trace1 = {
  "x": [5.0, 5.0, 15.0, 15.0],
  "y": [0.0, 1944.21269413, 1944.21269413, 0.0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace2 = {
  "x": [10.0, 10.0, 35.0, 35.0],
  "y": [1944.21269413, 1976.6580382, 1976.6580382, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace3 = {
  "x": [22.5, 22.5, 55.0, 55.0],
  "y": [1976.6580382, 1993.71562666, 1993.71562666, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace4 = {
  "x": [38.75, 38.75, 75.0, 75.0],
  "y": [1993.71562666, 2015.57014266, 2015.57014266, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace5 = {
  "x": [56.875, 56.875, 95.0, 95.0],
  "y": [2015.57014266, 2038.74127834, 2038.74127834, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace6 = {
  "x": [75.9375, 75.9375, 115.0, 115.0],
  "y": [2038.74127834, 2039.14565443, 2039.14565443, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace7 = {
  "x": [95.46875, 95.46875, 135.0, 135.0],
  "y": [2039.14565443, 2077.67490238, 2077.67490238, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace8 = {
  "x": [115.23, 115.23, 155.0, 155.0],
  "y": [2077.67490238, 2114.781549, 2114.781549, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace9 = {
  "x": [135.12, 135.12, 175.0, 175.0],
  "y": [2114.781549, 2152.28994329, 2152.28994329, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace10 = {
  "x": [165.09, 165.09, 195.0, 195.0],
  "y": [2152.28994329, 2168.80035965, 2168.80035965, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace11 = {
  "x": [180.092, 180.092, 215.0, 215.0],
  "y": [2168.80035965, 2200.34133716, 2200.34133716, 0],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
data = Data([trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11])
layout = {
  "autosize": False,
  "height": 500,
  "hovermode": "closest",
  "showlegend": False,
  "width": 800,
  "xaxis": {
    "mirror": "allticks",
    "rangemode": "tozero",
    "title": "Cluster Number",
    "showgrid": False,
    "showline": True,
    "showticklabels": True,
    "tickmode": "array",
    "ticks": "outside",
    "ticktext": ["288", "5", "41",  "286", "134", "34", "123", "29", "122",  "58",  "91",  "56"],
    "tickvals": [5.0,   15.0, 35.0, 55.0,  75.0,  95.0, 115.0, 135.0, 155.0, 175.0, 195.0, 215.0],
    "type": "linear",
    "zeroline": False
  },
  "yaxis": {
    "mirror": "allticks",
    "rangemode": "tozero",
    "showgrid": False,
    "showline": True,
    "title": "Distance",
    "showticklabels": True,
    "ticks": "outside",
    "type": "linear",
    "zeroline": False
  }
}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
