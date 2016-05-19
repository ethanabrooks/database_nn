import numpy as np
from bokeh.plotting import output_file, show, figure

x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)

output_file("legend.html")

p = figure()

kwargs_list = [
    {'legend': "sin(x)"},
    {'legend': "sin(x)",
     "line_dash": [4, 2],
     "line_color": "orange",
     "line_width": 2},
    {'legend': "3*sin(x)",
     "line_dash": [2, 4],
     "line_color": "green"}
]
dic = kwargs_list[0]
dic.update(kwargs_list[1])
print(dic)
