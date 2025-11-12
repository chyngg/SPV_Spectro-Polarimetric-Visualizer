import dearpygui.dearpygui as dpg
from matplotlib.colors import LinearSegmentedColormap

def setup_menu_themes():
	with dpg.theme() as menu_theme:
		with dpg.theme_component(dpg.mvButton):
			dpg.add_theme_color(dpg.mvThemeCol_Button, (252, 186, 3))
			dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 220, 50))
			dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255))
			dpg.add_theme_color(dpg.mvThemeCol_Text, (10, 10, 10))
	return menu_theme

def setup_tools_themes():
	with dpg.theme() as tools_theme:
		with dpg.theme_component(dpg.mvButton):
			dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 255, 110))
			dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 255, 160))
			dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255))
			dpg.add_theme_color(dpg.mvThemeCol_Text, (10, 10, 10))
		with dpg.theme_component(dpg.mvButton, enabled_state=False):
			dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 255, 110, 120))  # 살짝 투명
			dpg.add_theme_color(dpg.mvThemeCol_Text, (10, 10, 10, 180))
	return tools_theme

def setup_status_themes():
	with dpg.theme() as status_theme:
		with dpg.theme_component(dpg.mvText):
			dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 100, 100))
	return status_theme

def setup_fonts():
	with dpg.font_registry():
		dpg.add_font("assets/font/FreeSansBold.ttf", 18, tag="default_font")
	dpg.bind_font("default_font")

def make_custom_seismic():
	colors = [(0, 0, 1), (0, 0, 0.5), (0, 0, 0), (0.5, 0.5, 0), (1, 1, 0)]  # Blue -> Yellow
	positions = [0.0, 0.25, 0.5, 0.75, 1.0]  # Color positions
	custom_seismic = LinearSegmentedColormap.from_list("BlueYellow", list(zip(positions, colors)))
	return custom_seismic
