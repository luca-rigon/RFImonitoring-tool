import numpy as np
import matplotlib.pyplot as plt
import contextily as cx

def load_map(box_coordinates, azimuth, elevation):

    # Visualize map of the observatory location with drawn directions towards the main disturbances
    # Inputs: - coordinates in format (W, S, E, N)
    #       - azimuth (radians), elevation (degrees)

    # muc = cx.Place('Wettzell', zoom=12)
    # map_image = muc.im

    west, south, east, north = box_coordinates
    map_image, bbox = cx.bounds2img(west, south, east, north, ll=True, source=cx.providers.OpenStreetMap.Mapnik)

    (width, height, _) = map_image.shape
    scale = np.min([width,height])/2

    # Convert azimuth, elevation to cartesian coordinates (row,column):
    r = elevation*scale/90          # scale radius: ->>> consider zoom-factor
    theta = azimuth - np.pi/2

    row = r*np.cos(theta)
    col = r*np.sin(theta)

    # image center:
    row_0 = (height - 1) / 2
    col_0 = (width - 1) / 2

    delta = scale/3

    # Plot map, arrows & labels:
    plt.figure(figsize=(8, 8))
    plt.imshow(map_image, aspect='auto')
    plt.arrow(row_0-delta, col_0, 2*delta, 0, width=0.5, head_width=0.5)
    plt.arrow(row_0, col_0-delta, 0, 2*delta, width=0.5, head_width=0.5)
    plt.annotate('N', (row_0-12.5, col_0-delta-5))
    plt.annotate('S', (row_0-10, col_0+delta+30))
    plt.annotate('W', (row_0-delta-35, col_0+10))
    plt.annotate('E', (row_0+delta+5, col_0+10))
    plt.arrow(row_0, col_0, row, col, width=5, head_width=15, color='red')
    plt.axis('off')
    plt.show()
