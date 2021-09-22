import time

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

import analyse


def plot_network(
    env,
    t_index,
    height=400,
    width=400,
    cmap=None,
    save_to_file=False,
    alternative_data=None,
):
    """Return image of the network state at time given by index t_index.
    The color of the lines between the nodes is a measure for the amount of cars currently on them.
    There are a few ways to adjust the output mainly by adjusting the variables below.

    Keyword arguments:
    env -- simulation environment
    t_index -- time at which we screenshot the network
    height -- figure height
    width -- figure width
    cmap -- colormap used to color streets given their load
    save_to_file -- if True, a pdf file of the image is generated
    alternative_data -- if None, the data used to color the network is the number of cars
    """

    ################ overall settings ####################
    background_color = (255, 255, 255)

    # beauty settings
    circle_diameter = 20
    car_on_street_scale = 2
    edge_spacing = 5

    # colors
    circle_color = (255, 255, 255)
    circle_outline = (19, 114, 186)
    border_width_circle = 2

    line_there_color = (255, 187, 77)
    line_there_outline = (222, 140, 40)

    line_back_color = (174, 222, 98)
    line_back_outline = (120, 179, 48)
    border_width = 2

    # main image
    im = Image.new("RGB", (height, width), background_color)
    draw = ImageDraw.Draw(im)

    scale = (height - circle_diameter) / (env.network.n_x - 1)

    if alternative_data is None:
        data_to_map = env.state[t_index]
    else:
        data_to_map = alternative_data

    node_positions = env.network.node_positions()

    # draw all edges between nodes
    for i, x in enumerate(data_to_map[:]):
        (start, end) = env.network.edges[i]
        start_pos = node_positions[start]
        end_pos = node_positions[end]
        # get color of streets from street loads
        if cmap:
            line_there_color = line_back_color = cmap(x)
            line_back_outline = line_there_outline = line_there_color

        if end == start - env.network.n_x:  # draw lanes going up
            print((start, end), " goes up")
            start_pos = np.array(start_pos) * scale
            start_pos[0] += circle_diameter / 2 + edge_spacing
            start_pos[1] = height - start_pos[1] - circle_diameter / 2
            end_pos = np.array(end_pos) * scale
            end_pos[0] += circle_diameter / 2 + edge_spacing
            end_pos[0] += 3 * car_on_street_scale
            end_pos[1] = height - end_pos[1] - circle_diameter / 2
            line_coords = list(start_pos) + list(end_pos)
            draw.rectangle(
                line_coords,
                fill=line_there_color,
                outline=line_there_outline,
                width=border_width,
            )

        elif end == start + env.network.n_x:  # draw lanes going down
            print((start, end), " goes down")
            start_pos = np.array(start_pos) * scale
            start_pos[0] += circle_diameter / 2 - edge_spacing
            start_pos[1] = height - start_pos[1] - circle_diameter / 2
            end_pos = np.array(end_pos) * scale
            end_pos[0] += circle_diameter / 2 - edge_spacing
            end_pos[0] -= 3 * car_on_street_scale
            end_pos[1] = height - end_pos[1] - circle_diameter / 2
            line_coords = list(start_pos) + list(end_pos)
            draw.rectangle(
                line_coords,
                fill=line_back_color,
                outline=line_back_outline,
                width=border_width,
            )

        elif end == start + 1:  # draw lanes going right
            print((start, end), " goes right")
            start_pos = np.array(start_pos) * scale
            start_pos[0] += circle_diameter / 2
            start_pos[1] = height - start_pos[1] - circle_diameter / 2 + edge_spacing
            end_pos = np.array(end_pos) * scale
            end_pos[0] += circle_diameter / 2
            end_pos[1] = height - end_pos[1] - circle_diameter / 2 + edge_spacing
            end_pos[1] += 3 * car_on_street_scale
            line_coords = list(start_pos) + list(end_pos)
            draw.rectangle(
                line_coords,
                fill=line_there_color,
                outline=line_there_outline,
                width=border_width,
            )

        elif end == start - 1:  # draw lanes going left
            print((start, end), " goes left")
            start_pos = np.array(start_pos) * scale
            start_pos[0] += circle_diameter / 2
            start_pos[1] = height - start_pos[1] - circle_diameter / 2 - edge_spacing
            end_pos = np.array(end_pos) * scale
            end_pos[0] += circle_diameter / 2
            end_pos[1] = height - end_pos[1] - circle_diameter / 2 - edge_spacing
            end_pos[1] -= 3 * car_on_street_scale
            line_coords = list(start_pos) + list(end_pos)
            draw.rectangle(
                line_coords,
                fill=line_back_color,
                outline=line_back_outline,
                width=border_width,
            )

    # draw nodes over edges
    for i, x in enumerate(env.network.edges):
        bottom_left = np.array(node_positions[x[0]]) * scale
        bottom_left[1] = height - bottom_left[1] - circle_diameter
        bounding_box = list(bottom_left) + list(bottom_left + circle_diameter)
        draw.ellipse(
            tuple(bounding_box),
            fill=circle_color,
            outline=circle_outline,
            width=border_width_circle,
        )

    if save_to_file:
        im.save(f"{time.time()}.pdf", quality=100)
    return im


def draw_car_distribution(env):
    """Plot the network; edge colors indicate average number of cars. Save image to pdf file."""
    cars = analyse.avg_cars_streetwise(env, data_type="environment")
    max_N = 16
    colors = plt.get_cmap("viridis").colors

    def cmap(N):
        index = int((N / max_N) * 256)
        index = 255 if index > 255 else index
        rgba = colors[index]
        return tuple(int(v * 256) for v in rgba)

    im = plot_network(env, 0, save_to_file=True, cmap=cmap, alternative_data=cars)

    return im
