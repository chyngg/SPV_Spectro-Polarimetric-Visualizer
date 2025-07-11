import dearpygui.dearpygui as dpg
import dearpygui_extend as dpge
from subs import *
import numpy as np

current_tab = "Trichromatic"
selectable_options = ["s0", "s1", "s2", "s3", "DoLP", "AoLP", "DoCP", "CoP", "Unpolarized", "Polarized(Linear)", "Polarized(Circular)", "Polarized(total)"]
wavelength_options = ["R", "G", "B"]
graph_options = ["s0", "s1", "s2", "s3", "DoLP", "AoLP", "DoCP", "CoP"]

dpg.create_context()

setup_fonts()
layout= ('\n'
		 'LAYOUT example\n'
		 '	COL left_menu 0.2 left top\n'
		 '	COL content 0.6\n'
		 '		ROW 0.7\n'
		 '			COL center_img 1 center center\n'
		 '		ROW 0.3\n'
		 '			COL nodes_cat_A\n'
		 '			COL nodes_cat_B\n'
		 '	COL tools 0.2 center top\n')

with dpg.window(tag='main_window'):
	dpge.add_layout(layout, debug=False, resizable=False, border=True)

# themes
with dpg.theme() as menu_theme:
	with dpg.theme_component(dpg.mvButton):
		dpg.add_theme_color(dpg.mvThemeCol_Button, (252, 186, 3))
		dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 220, 50))
		dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255))
		dpg.add_theme_color(dpg.mvThemeCol_Text, (10, 10, 10))
with dpg.theme() as tools_theme:
	with dpg.theme_component(dpg.mvButton):
		dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 255, 110))
		dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 255, 160))
		dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255))
		dpg.add_theme_color(dpg.mvThemeCol_Text, (10, 10, 10))
with dpg.theme() as status_theme:
	with dpg.theme_component(dpg.mvText):
		dpg.add_theme_color(dpg.mvThemeCol_Text, (100,100,100))

# LEFT MENU
with dpg.group(parent='left_menu', indent=5) as left_menu:
	dpg.add_text('MENU:')
	dpg.add_button(label="Upload .npy File", callback=open_import_dialog, tag="upload_file_button", width=-1)
	dpg.add_button(label="Export image", callback=open_export_dialog, tag="export_image_button", width=-1)

	dpg.add_spacer(height=30)
	dpg.add_separator()
	dpg.add_spacer(height=5)
	dpg.add_button(label="Distribution of Stokes vector", callback=lambda: show_stokes_histogram(1), width=-1)
	dpg.add_spacer(height=5)
	dpg.add_separator()
	dpg.add_button(label="Gradient distributions of\n\t  Stokes vector", callback=lambda: show_histogram_callback(2),
				   width=-1)
	dpg.add_button(label="Gradient distributions of\nStokes vector derivatives",
				   callback=lambda: show_histogram_callback(3), width=-1)
	dpg.add_button(label="Distributions of polarization features", callback=lambda: show_histogram_callback(4),
				   width=-1)
	with dpg.group(horizontal=True):
		dpg.add_text('Select Direction: ')
		dpg.add_radio_button(items=["X", "Y"], callback=change_direction, tag="Direction", default_value="X",
							 horizontal=True)
	dpg.add_spacer(height=5)
	dpg.add_separator()
dpg.bind_item_theme(left_menu, menu_theme)

# TOOLS RIGHT MENU
with dpg.group(parent='tools', indent=5) as tools:
	dpg.add_text('Visualize by entire wavelengths:')
	dpg.add_button(label='original', callback=lambda: set_selected_option("original"), width=-1)
	dpg.add_button(label='s0', callback=lambda: set_selected_option("s0"), width=-1)
	dpg.add_button(label='s1', callback=lambda: set_selected_option("s1"), width=-1)
	dpg.add_button(label='s2', callback=lambda: set_selected_option("s2"), width=-1)
	dpg.add_button(label='s3', callback=lambda: set_selected_option("s3"), width=-1)
	dpg.add_button(label='DoLP', callback=lambda: set_selected_option("DoLP"), width=-1)
	dpg.add_button(label='AoLP', callback=lambda: set_selected_option("AoLP"), width=-1)
	dpg.add_button(label='DoCP', callback=lambda: set_selected_option("DoCP"), width=-1)
	dpg.add_button(label='CoP', callback=lambda: set_selected_option("CoP"), width=-1)
	dpg.add_button(label='Unpolarized', callback=lambda: set_selected_option("Unpolarized"), width=-1)
	dpg.add_button(label='Polarized (Linear)', callback=lambda: set_selected_option("Polarized (linear)"), width=-1)
	dpg.add_button(label='Polarized (Circular)', callback=lambda: set_selected_option("Polarized (circular)"), width=-1)
	dpg.add_button(label='Polarized (total)', callback=lambda: set_selected_option("Polarized (total)"), width=-1)
	dpg.add_spacer(height=10)
	dpg.add_separator()

	dpg.add_text('Visualize by Individual Wavelength: ')
	dpg.add_button(label="Visualize", callback=lambda: activate_visualization(), width=-1)
	dpg.add_text('Select Visualization option:')
	dpg.add_combo(items=selectable_options, default_value="s0", tag="polarimetric_options", callback=select_option_callback, enabled=False, width=-1)
	dpg.add_text('Select Wavelength:')
	dpg.add_combo(items=wavelength_options, default_value=wavelength_options[0], tag="wavelength_options", callback=select_option_callback, enabled=False, width=-1)

dpg.bind_item_theme(tools, tools_theme)

# NODES CATEGORY A
with dpg.group(parent='nodes_cat_A', indent=20) as ndo_cat_A:
	dpg.add_spacer(height=10)
	dpg.add_text("Enter Coordinates:")
	dpg.add_separator()
	with dpg.group(horizontal=True):
		dpg.add_text("Upper Left    |  ")
		dpg.add_text("X: ")
		dpg.add_input_text(tag="upper_left_x", callback=on_upper_left_x, width=100)
		dpg.add_text("Y: ")
		dpg.add_input_text(tag="upper_left_y", callback=on_upper_left_y, width=100)
	with dpg.group(horizontal=True):
		dpg.add_text("Lower Right |  ")
		dpg.add_text("X: ")
		dpg.add_input_text(tag="lower_right_x", callback=on_lower_right_x, width=100)
		dpg.add_text("Y: ")
		dpg.add_input_text(tag="lower_right_y", callback=on_lower_right_y, width=100)
	dpg.add_separator()
	dpg.add_text("Select graph option:")
	dpg.add_combo(items=graph_options, default_value="s0", tag="graph_options",
				  callback=graph_option_callback, width=-1)
	dpg.add_separator()
	dpg.add_button(label="View Graph", callback=lambda: view_graph(), width=-1)

# NODES CATEGORY B
with dpg.group(parent='nodes_cat_B', indent=20) as ndo_cat_B:
	dpg.add_spacer(height=10)
	dpg.add_text("Enter vmax:\n")
	dpg.add_input_text(tag="vmax_input", callback=on_vmax_change)
	dpg.add_spacer(height=10)
	dpg.add_text("Enter vmin:\n")
	dpg.add_input_text(tag="vmin_input", callback=on_vmin_change)
	dpg.add_spacer(height=10)
	with dpg.group(horizontal=True):
		dpg.add_button(label="Update Visualization", callback=reload_visualization, width=190)
		dpg.add_button(label="Reset Visualization", callback=reset_visualization, width=190)


# Center
with dpg.group(parent='center_img', indent=5) as center_img:
	with dpg.texture_registry(show=False):
		dpg.add_dynamic_texture(760, 540, np.zeros((760, 540, 4), dtype=np.float32).flatten(), tag="uploaded_texture")
	with dpg.child_window(width=760, height=540, tag="uploaded_image"):
		dpg.add_image("uploaded_texture", width=760, height=540)

dpg.create_viewport(title='Spectro-polarimetric GUI')
dpg.set_primary_window('main_window', True)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()