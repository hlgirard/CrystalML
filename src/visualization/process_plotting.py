import os

from plotly import offline as py
from plotly import tools
import plotly.graph_objs as go
import plotly.io as pio

def plot_crystal_data(df, directory):
    '''Plot data from crystallization experiment'''

    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Crystallization kinetics', 'Process control'))

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num drops"],
        name = 'Total',
        hoverinfo = 'text',
        text = df["Image Name"]
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num clear"],
        name = 'Clear'
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num crystal"],
        name = 'Crystal'
    ), 2, 1)

    fig.append_trace(go.Scatter(
        x = df["RelTime"],
        y = df["Num clear"] / df["Num drops"],
        name = 'Clear/Total'
    ), 1, 1)

    fig['layout']['xaxis'].update(title='Time (s)')
    fig['layout']['yaxis2'].update(title='Number of drops')
    fig['layout']['yaxis1'].update(title='Clear/Total', range=[0,1.05])

    fig['layout'].update(
        title='Crystallization kinetics data for {}'.format(directory)
    )

    py.plot(fig)
    pio.write_image(fig, os.path.join(directory, 'Crystallization_kinetics_plot.pdf'))
