import plotly.plotly as py
from plotly.graph_objs import *

trace1 = {
  "x": [5.0, 5.0, 15.0, 15.0],
  "y": [0.0, 3051.97755562, 3051.97755562, 0.0],
  "marker": {"color": "rgb(255,51,51)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace2 = {
  "x": [25.0, 25.0, 35.0, 35.0],
  "y": [0.0, 2706.01662966, 2706.01662966, 0.0],
  "marker": {"color": "rgb(255,128,0)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace3 = {
  "x": [45.0, 45.0, 55.0, 55.0],
  "y": [0.0, 2684.97094956, 2684.97094956, 0.0],
  "marker": {"color": "rgb(255,255,0)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace4 = {
  "x": [65.0, 65.0, 75.0, 75.0],
  "y": [0.0, 2780.74036904, 2780.74036904, 0.0],
  "marker": {"color": "rgb(128,255,0)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace5 = {
  "x": [70.0, 70.0, 95.0, 95.0],
  "y": [2780.74036904, 3003.82722539, 3003.82722539, 0.0],
  "marker": {"color": "rgb(0,153,0)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace6 = {
  "x": [105.0, 105.0, 130.0, 130.0],
  "y": [0.0, 2900.29601937, 2900.29601937, 2887.18513435],
  "marker": {"color": "rgb(0,255,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace7 = {
  "x": [125.0, 125.0, 135.0, 135.0],
  "y": [0.0, 2887.18513435, 2887.18513435, 0.0],
  "marker": {"color": "rgb(61,153,112)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace8 = {
  "x": [10.0, 10.0, 30.0, 30.0],
  "y": [3051.97755562, 3086.77938959, 3086.77938959, 2706.01662966],
  "marker": {"color": "rgb(0,128,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace9 = {
  "x": [82.5, 82.5, 117.5, 117.5],
  "y": [3003.82722539, 3144.01781165, 3144.01781165, 2900.29601937],
  "marker": {"color": "rgb(127,0,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace10 = {
  "x": [50, 50, 100, 100],
  "y": [2684.97094956, 3335.61103848, 3335.61103848, 3144.01781165],
  "marker": {"color": "rgb(255,0,255)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace11 = {
  "x": [20, 20, 75, 75],
  "y": [3086.77938959, 3512.02135529, 3512.02135529, 3335.61103848],
  "marker": {"color": "rgb(255,51,153)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
data = Data([trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11])
layout = {
  "autosize": True,
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
    "ticktext": ["286",  "268", "277", "251", "249", "56", "285", "278", "288", "283", "282", "287"],
    "tickvals": [5.0,    15.0,  25.0,  35.0,  45.0,  55.0, 65.0,  75.0,  95.0,  105.0, 125.0, 135.0],
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
