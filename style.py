import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({'axes.linewidth' : 1.5, 
                     'ytick.major.width' : 1.5,
                     'ytick.minor.width' : 1.5,
                     'xtick.major.width' : 1.5,
                     'xtick.minor.width' : 1.5,
                     'xtick.labelsize': 14, 
                     'ytick.labelsize': 14,
                     'axes.labelsize': 24/2,
                     'axes.labelpad' : 2,
                     'axes.titlesize' : 36/2,
                     'axes.titlepad' : 10/2,
                     'font.family': 'Serif'
                    })

mpl.rcParams['mathtext.fontset'] = 'cm'          # Computer Modern serif
mpl.rcParams['mathtext.rm'] = 'serif'

plot_colors_rgb = [
    (0/255, 107/255, 164/255),   # 006BA4
    (255/255, 128/255, 14/255),  # FF800E
    (171/255, 171/255, 171/255), # ABABAB
    (89/255, 89/255, 89/255),    # 595959
    (95/255, 158/255, 209/255),  # 5F9ED1
    (200/255, 82/255, 0/255),    # C85200
    (137/255, 137/255, 137/255), # 898989
    (162/255, 200/255, 236/255), # A2C8EC
    (255/255, 188/255, 121/255), # FFBC79
    (207/255, 207/255, 207/255)  # CFCFCF
]