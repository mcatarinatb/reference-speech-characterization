import os
import numpy as np
 
SEED=741
np.random.seed(seed=SEED)

import plotly.graph_objects as go
import matplotlib.pyplot as plt



def radar_plot(
    categories, values_to_plot=[], sub_plot_legend=["adress", "daic"], 
    colors=[], plot_path="radar_plot.png",
    return_fig=True
):

    numeric_categories = [str(s) for s in np.arange(len(categories))]
    fig = go.Figure()

    # plot each subplot 
    for values, sub_plot, color in zip(values_to_plot, sub_plot_legend, colors):
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=numeric_categories,
            #fill='toself',
            name=sub_plot,
            line_color=color,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                # todo - find a smart way to define the range
                #range=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] # axis range
                range = (-4., 5.)
                ),
        ),
        font = dict(size=16),
        #showlegend=False,
        width=1000, 
        height=1000
    )
    fig.show()
    
    # print categories names, because printing the categories names in the radial
    # axis was not possible. names have too many characters.
    for i, c in enumerate(categories):
        print(i, "-", c)
    
    if return_fig:
        return fig
    