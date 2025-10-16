import matplotlib.pyplot as plt

''''
This file contains configurations relevant to the creation of figures.

'''

def load_general_styles():
    ''' this function is used to have uniform plotting styles between the different postprocessing notebooks in this project
        plotting styles are set in a dict and then returned '''

    general_styles = {
    'palette_box_plot': 'colorblind',   
    'palette_light': ['#7BC1E8', '#E69B63'],        # pallete for seaborn box plots
    'palette_dark': 'dark',                             # pallete for seaborn box plots
    'column_width': 17.4,                               # column width in cm 
    'colours': {'blue': (0, 0.6, 1., 0.7), 
                'orange': (1, 0.5, 0., 0.7)}
    }

    return general_styles

def load_gravitational_waves_styles():
    ''' this function is used to have uniform plotting styles between the different postprocessing notebooks in this project
        plotting styles are set in a dict and then returned'''

    gravitational_waves_styles = {
    'class_labels': ['NS - BH / NS - NS', 'Small BH - Small BH', 'Large BH - Large BH '],
    'class_colours': ['tab:blue', 'tab:orange', 'tab:green']
    }

    return gravitational_waves_styles

def load_prednisone_3D_styles():
    ''' this function is used to have uniform plotting styles between the different postprocessing notebooks in this project
        plotting styles are set in a dict and then returned'''

    prednisone_3D_styles = {
    'class_labels': ['class 0', 'class 1'],
    'class_colours': ['tab:blue', 'tab:orange'],
    'class_markers': ['*', '.'],
    }

    plt.style.use('prednisone.mplstyle')

    return prednisone_3D_styles

def load_restricted_unrestricted_styles():
    restricted_unrestricted = {
    'hue_order': ['spherical_restricted', 'spherical_unrestricted', 'full_restricted', 'full_unrestricted'],
    'palette': [(0, 0.6, 1, 0.5), (0, 0.6, 1., 0.7), (1, 0.5, 0., 0.5), (1, 0.5, 0., 0.7)],
    }

    return restricted_unrestricted