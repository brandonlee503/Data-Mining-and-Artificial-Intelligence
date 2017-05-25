import plotly.plotly as py
from plotly.graph_objs import *

trace1 = {
  "x": [15.0, 15.0, 25.0, 25.0],
  "y": [0.0, 2208.69667451, 2208.69667451, 0.0],
  "marker": {"color": "rgb(61,153,112)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace2 = {
  "x": [5.0, 5.0, 20.0, 20.0],
  "y": [0.0, 2433.51104374, 2433.51104374, 2208.69667451],
  "marker": {"color": "rgb(0,116,217)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace3 = {
  "x": [65.0, 65.0, 75.0, 75.0],
  "y": [0.0, 1696.82301193, 1696.82301193, 0.0],
  "marker": {"color": "rgb(255,65,54)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace4 = {
  "x": [55.0, 55.0, 70.0, 70.0],
  "y": [0.0, 2068.32888076, 2068.32888076, 1696.82301193],
  "marker": {"color": "rgb(255,65,54)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace5 = {
  "x": [45.0, 45.0, 62.5, 62.5],
  "y": [0.0, 2263.29450139, 2263.29450139, 2068.32888076],
  "marker": {"color": "rgb(255,65,54)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace6 = {
  "x": [35.0, 35.0, 53.75, 53.75],
  "y": [0.0, 2496.30526979, 2496.30526979, 2263.29450139],
  "marker": {"color": "rgb(0,116,217)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace7 = {
  "x": [12.5, 12.5, 44.375, 44.375],
  "y": [2433.51104374, 2893.68467529, 2893.68467529, 2496.30526979],
  "marker": {"color": "rgb(0,116,217)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace8 = {
  "x": [95.0, 95.0, 105.0, 105.0],
  "y": [0.0, 2624.00285823, 2624.00285823, 0.0],
  "marker": {"color": "rgb(0,116,217)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace9 = {
  "x": [85.0, 85.0, 100.0, 100.0],
  "y": [0.0, 3015.90931561, 3015.90931561, 2624.00285823],
  "marker": {"color": "rgb(0,116,217)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
trace10 = {
  "x": [28.4375, 28.4375, 92.5, 92.5],
  "y": [2893.68467529, 3234.17160955, 3234.17160955, 3015.90931561],
  "marker": {"color": "rgb(0,116,217)"},
  "mode": "lines",
  "type": "scatter",
  "xaxis": "x",
  "yaxis": "y"
}
data = Data([trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10])
layout = {
  "autosize": False,
  "height": 500,
  "hovermode": "closest",
  "showlegend": False,
  "width": 800,
  "xaxis": {
    "mirror": "allticks",
    "rangemode": "tozero",
    "showgrid": False,
    "showline": True,
    "showticklabels": True,
    "tickmode": "array",
    "ticks": "outside",
    "ticktext": ["2", "7", "8", "5", "0", "1", "9", "10", "3", "4", "6"],
    "tickvals": [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0],
    "type": "linear",
    "zeroline": False
  },
  "yaxis": {
    "mirror": "allticks",
    "rangemode": "tozero",
    "showgrid": False,
    "showline": True,
    "showticklabels": True,
    "ticks": "outside",
    "type": "linear",
    "zeroline": False
  }
}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
