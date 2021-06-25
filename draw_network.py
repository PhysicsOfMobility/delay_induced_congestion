import time

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import analyse


def plot_network(
    env, t_index, heigth=400, width=400, padding=1.1, cmap=None, save_to_file=False, alternative_data=None, comment=""
):
    """returns image of the network state at time given by index t_index. The width of the lines between
    the nodes is a measure for the amount of cars currently on them. There are a few ways to adjust the output
    mainly by adjusting the variables below. The kwargs are used for some external control... no matter what you do. do not try to
    read this code. It is seriously messed up."""

    ################ overall settings ####################
    background_color = (255, 255, 255)
    font_size = 20

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

    ########## read till her. Do not venture deeper into this function. For the code is dark and treacherous.########
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    # main image
    im = Image.new("RGB", (heigth, width), background_color)
    draw = ImageDraw.Draw(im)

    scale = (heigth - circle_diameter) / (env.network.n_x - 1)

    if alternative_data is None:
        data_to_map = env.state[t_index]
    else:
        data_to_map = alternative_data

    # draw all edges between nodes
    for i, x in enumerate(data_to_map):
        try:  # if street is None, dont draw an edge
            start, end = env.network.get_coords_from_id(i)
        except ValueError:
            continue

        if cmap:
            line_there_color = line_back_color = cmap(x)
            line_back_outline = line_there_outline = line_there_color

        if i % 4 == 0:  # draw lanes going up
            start = np.array(start) * scale
            start[0] += circle_diameter / 2 + edge_spacing
            start[1] = heigth - start[1] - circle_diameter / 2
            end = np.array(end) * scale
            end[0] += circle_diameter / 2 + edge_spacing
            end[0] += 3 * car_on_street_scale
            end[1] = heigth - end[1] - circle_diameter / 2
            line_coords = list(start) + list(end)
            draw.rectangle(line_coords, fill=line_there_color, outline=line_there_outline, width=border_width)

        elif i % 4 == 2:  # draw lanes going down
            start = np.array(start) * scale
            start[0] += circle_diameter / 2 - edge_spacing
            start[1] = heigth - start[1] - circle_diameter / 2
            end = np.array(end) * scale
            end[0] += circle_diameter / 2 - edge_spacing
            end[0] -= 3 * car_on_street_scale
            end[1] = heigth - end[1] - circle_diameter / 2
            line_coords = list(start) + list(end)
            draw.rectangle(line_coords, fill=line_back_color, outline=line_back_outline, width=border_width)

        elif i % 4 == 3:  # draw lanes going right
            start = np.array(start) * scale
            start[0] += circle_diameter / 2
            start[1] = heigth - start[1] - circle_diameter / 2 + edge_spacing
            end = np.array(end) * scale
            end[0] += circle_diameter / 2
            end[1] = heigth - end[1] - circle_diameter / 2 + edge_spacing
            end[1] += 3 * car_on_street_scale
            line_coords = list(start) + list(end)
            draw.rectangle(line_coords, fill=line_there_color, outline=line_there_outline, width=border_width)

        elif i % 4 == 1:  # draw lanes going left
            start = np.array(start) * scale
            start[0] += circle_diameter / 2
            start[1] = heigth - start[1] - circle_diameter / 2 - edge_spacing
            end = np.array(end) * scale
            end[0] += circle_diameter / 2
            end[1] = heigth - end[1] - circle_diameter / 2 - edge_spacing
            end[1] -= 3 * car_on_street_scale
            line_coords = list(start) + list(end)
            draw.rectangle(line_coords, fill=line_back_color, outline=line_back_outline, width=border_width)

    # draw nodes over edges
    for i, x in enumerate(env.network.streets):
        bottom_left = env.network.get_point_from_pointid(i) * scale
        bottom_left[1] = heigth - bottom_left[1] - circle_diameter
        bounding_box = list(bottom_left) + list(bottom_left + circle_diameter)
        draw.ellipse(tuple(bounding_box), fill=circle_color, outline=circle_outline, width=border_width_circle)

    if save_to_file:
        im.save(f"{time.time()}.pdf", quality=100)
    return im


def save_anim_network(env, name="out", timestep=400, colormap=True):
    """saves an animated gif of the network attached to env."""
    if colormap:
        max_N = np.max(env.state)

        colors = plt.get_cmap("viridis").colors

        def cmap(N):
            index = int(N / max_N * 256)
            index = 255 if index > 255 else index
            rgba = colors[index]
            return tuple(int(v * 256) for v in rgba)

    images = []
    for i in range(env.state.shape[0]):
        images.append(plot_network(env, i, cmap=cmap))
    images[0].save(
        f"{name}_{time.time()}.gif",
        save_all=True,
        append_images=images[1:],
        duration=timestep * env.state_time_resolution,
        loop=0,
    )


def draw_car_distribution(env_state, env):
    cars = analyse.avg_cars_streetwise(env_state, data_type="statevector")
    max_N = 16
    colors = plt.get_cmap("viridis").colors

    def cmap(N):
        index = int((N / max_N) * 256)
        index = 255 if index > 255 else index
        rgba = colors[index]
        return tuple(int(v * 256) for v in rgba)

    im = plot_network(env, 0, save_to_file=True, cmap=cmap, alternative_data=cars)

    return im
