import dearpygui.dearpygui as dpg
from matplotlib.colors import LinearSegmentedColormap

def setup_fonts():
	with dpg.font_registry():
		dpg.add_font("assets/font/FreeSansBold.ttf", 18, tag="default_font")
	dpg.bind_font("default_font")

def make_custom_seismic():
	colors = [(0, 0, 1), (0, 0, 0.5), (0, 0, 0), (0.5, 0.5, 0), (1, 1, 0)]  # Blue -> Yellow
	positions = [0.0, 0.25, 0.5, 0.75, 1.0]  # Color positions
	custom_seismic = LinearSegmentedColormap.from_list("BlueYellow", list(zip(positions, colors)))
	return custom_seismic
