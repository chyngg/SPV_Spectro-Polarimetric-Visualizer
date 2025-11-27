import dearpygui_extend as dpge
import dearpygui.dearpygui as dpg
from subs import *
import numpy as np

dpg.create_context()
history_db.init_history_db()

themes.setup_fonts()
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
menu_theme = themes.setup_menu_themes()
tools_theme = themes.setup_tools_themes()
status_theme = themes.setup_status_themes()

# LEFT MENU
with dpg.group(parent='left_menu', indent=5) as left_menu:
	dpg.add_text('MENU:')
	dpg.add_button(label="Upload .npy File", callback=io_utils.open_import_dialog, tag="upload_file_button", width=-1)
	dpg.add_button(label="Export image", callback=io_utils.open_export_dialog, tag="export_image_button", width=-1)

	dpg.add_spacer(height=30)
	dpg.add_separator()
	dpg.add_spacer(height=5)
	dpg.add_button(label="Distribution of Stokes vector", callback=lambda: histogram.show_stokes_histogram(1), width=-1)
	dpg.add_spacer(height=5)
	dpg.add_separator()
	dpg.add_button(label="Gradient distributions of\n\t  Stokes vector",
				   callback=lambda: callbacks.show_histogram_callback(2), width=-1)
	dpg.add_button(label="Gradient distributions of\nStokes vector derivatives",
				   callback=lambda: callbacks.show_histogram_callback(3), width=-1)
	dpg.add_button(label="Distributions of polarization features",
				   callback=lambda: callbacks.show_histogram_callback(4), width=-1)
	with dpg.group(horizontal=True):
		dpg.add_text('Select Direction: ')
		dpg.add_radio_button(items=["X", "Y"], callback=callbacks.change_direction, tag="Direction",
							 default_value="X", horizontal=True)
	dpg.add_spacer(height=5)
	dpg.add_separator()
	dpg.add_text("History")
	with dpg.group(tag="history_group"):
		with dpg.child_window(tag="checkbox_area", height=250, parent="history_group", autosize_x=True, border=True):
			pass
		dpg.add_button(label="Load Single File", callback=io_utils.load_file_callback, parent="history_group", width=-1)
		dpg.add_separator(parent="history_group")
		dpg.add_button(label="Show graph for Selected Files", callback=graph.show_combined_graph,
					   parent="history_group", width=-1)
		dpg.add_combo(items=sp_state.graph_options, parent="history_group", default_value="s0",
					  tag="multi_graph_options", callback=callbacks.multi_graph_option_callback, width=-1)

dpg.bind_item_theme(left_menu, menu_theme)

# TOOLS RIGHT MENU
with dpg.group(parent='tools', indent=5) as tools:
	dpg.add_text('Visualize by entire wavelengths:')
	dpg.add_button(label='original', callback=lambda: callbacks.set_sp_option("original"), width=-1)
	dpg.add_button(label='s0', callback=lambda: callbacks.set_sp_option("s0"), width=-1)
	dpg.add_button(label='s1', callback=lambda: callbacks.set_sp_option("s1"), width=-1)
	dpg.add_button(label='s2', callback=lambda: callbacks.set_sp_option("s2"), width=-1)
	dpg.add_button(label='s3', callback=lambda: callbacks.set_sp_option("s3"), width=-1)
	dpg.add_button(label='DoLP', callback=lambda: callbacks.set_sp_option("DoLP"), width=-1)
	dpg.add_button(label='AoLP', callback=lambda: callbacks.set_sp_option("AoLP"), width=-1)
	dpg.add_button(label='DoCP', callback=lambda: callbacks.set_sp_option("DoCP"), width=-1)
	dpg.add_button(label='CoP', callback=lambda: callbacks.set_sp_option("CoP"), width=-1)
	dpg.add_button(label='Unpolarized', callback=lambda: callbacks.set_sp_option("Unpolarized"), width=-1)
	with dpg.group(horizontal=True):
		dpg.add_button(label='Polarized', callback=lambda: callbacks.set_sp_option(f"Polarized ({sp_state.now_polarized})"), width=160)
		dpg.add_combo(items=sp_state.polarized_options, callback=callbacks.change_polarized_option_callback, default_value="total",
					  tag="select_polarized_option", width=110)
	dpg.add_separator()

	dpg.add_text('Visualize by Individual Wavelength: ')
	dpg.add_button(label="Visualize", callback=lambda: callbacks.activate_visualization(), width=-1)
	with dpg.group(horizontal=True):
		dpg.add_text('Visualization option:')
		dpg.add_combo(items=sp_state.selectable_options, default_value="s0", tag="polarimetric_options",
					  callback=callbacks.set_sp_by_channel_callback, enabled=False, width=-1)
	with dpg.group(horizontal=True):
		dpg.add_text('Wavelength:')
		dpg.add_combo(items=common_state.wavelength_options, default_value=common_state.wavelength_options[0], tag="wavelength_options",
					  callback=callbacks.set_sp_by_channel_callback, enabled=False, width=-1)
	dpg.add_spacer(height=10)
	dpg.add_separator()

	dpg.add_text('For Mueller-matrix image: ')
	with dpg.group(horizontal=True):
		dpg.add_text('Correction: ')
		dpg.add_combo(items=mueller_state.corrections, default_value=mueller_state.corrections[0], tag="mueller_correction",
					  callback=callbacks.mueller_select_option_callback, enabled=False, width=-1)
	with dpg.group(horizontal=True):
		dpg.add_text('Gamma value: ')
		dpg.add_input_double(tag='gamma_input', default_value=2.2, callback=callbacks.on_gamma_change, width=170)
	dpg.add_separator()
	with dpg.group(horizontal=True):
		dpg.add_text('Channel: ')
		dpg.add_combo(items=common_state.wavelength_options, default_value="R", tag="mueller_channel",
					  callback=callbacks.mueller_channel_callback, enabled=False, width=-1)
	dpg.add_text('Visualize as RGB: ')
	with dpg.group(parent=tools, horizontal=True):
		dpg.add_button(label="Positive", callback=callbacks.mueller_rgb_callback_positive, tag="Mueller_rgb_positive", enabled=False, width=135)
		dpg.add_button(label="Negative", callback=callbacks.mueller_rgb_callback_negative, tag="Mueller_rgb_negative", enabled=False, width=135)

	dpg.add_separator()
	dpg.add_text("Mueller video controls:")
	with dpg.group(horizontal=True):
		dpg.add_button(
			label="Play",
			tag="mueller_video_play",
			width=70,
			enabled=False,
			callback=mueller_video.cb_play,
		)
		dpg.add_button(
			label="Pause",
			tag="mueller_video_pause",
			width=70,
			enabled=False,
			callback=mueller_video.cb_pause,
		)

	with dpg.group(horizontal=True):
		dpg.add_button(
			label="< Prev",
			tag="mueller_video_prev",
			width=70,
			enabled=False,
			callback=mueller_video.cb_prev,
		)
		dpg.add_button(
			label="Next >",
			tag="mueller_video_next",
			width=70,
			enabled=False,
			callback=mueller_video.cb_next,
		)

	with dpg.group(horizontal=True):
		dpg.add_text("FPS:")
		dpg.add_input_text(
			tag="mueller_video_fps",
			default_value="10",
			width=80,
			enabled=False,
			callback=mueller_video.cb_fps,
		)

dpg.bind_item_theme(tools, tools_theme)

# NODES CATEGORY A
with dpg.group(parent='nodes_cat_A', indent=20) as ndo_cat_A:
	dpg.add_spacer(height=5)
	dpg.add_text("Enter Coordinates:")
	dpg.add_separator()
	with dpg.group(horizontal=True):
		dpg.add_text("Lower Left    |  ")
		dpg.add_text("X: ")
		dpg.add_input_text(tag="lower_left_x", callback=callbacks.on_lower_left_x, width=70)
		dpg.add_text("Y: ")
		dpg.add_input_text(tag="lower_left_y", callback=callbacks.on_lower_left_y, width=70)
	with dpg.group(horizontal=True):
		dpg.add_text("Upper Right |  ")
		dpg.add_text("X: ")
		dpg.add_input_text(tag="upper_right_x", callback=callbacks.on_upper_right_x, width=70)
		dpg.add_text("Y: ")
		dpg.add_input_text(tag="upper_right_y", callback=callbacks.on_upper_right_y, width=70)
	dpg.add_separator()
	with dpg.group(horizontal=True):
		dpg.add_text("Select graph option:")
		dpg.add_combo(items=sp_state.graph_options, default_value="s0", tag="crop_graph_options",
					  callback=callbacks.crop_graph_option_callback, width=252)
	dpg.add_button(label="View Graph", callback=lambda: graph.view_graph(), width=400)

# NODES CATEGORY B
with dpg.group(parent='nodes_cat_B', indent=20) as ndo_cat_B:
	dpg.add_spacer(height=10)
	dpg.add_text("Enter vmax:\n")
	dpg.add_input_float(tag="vmax_input", callback=callbacks.on_vmax_change)
	dpg.add_spacer(height=10)
	dpg.add_text("Enter vmin:\n")
	dpg.add_input_float(tag="vmin_input", callback=callbacks.on_vmin_change)
	dpg.add_spacer(height=10)
	with dpg.group(horizontal=True):
		dpg.add_button(label="Update Visualization", callback=callbacks.reload_visualization, width=190)
		dpg.add_button(label="Reset Visualization", callback=callbacks.reset_visualization, width=190)

# Center
with dpg.group(parent='center_img', indent=5) as center_img:
	with dpg.texture_registry(show=False):
		dpg.add_dynamic_texture(760, 540, np.zeros((760, 540, 4), dtype=np.float32).flatten(), tag="uploaded_texture")
	with dpg.child_window(width=760, height=540, tag="uploaded_image"):
		dpg.add_image("uploaded_texture", width=760, height=540)
	dpg.add_slider_int(
		label="",
		tag="mueller_video_slider",
		min_value=0,
		max_value=1,
		default_value=0,
		width=760,
		enabled=False,
		show=False,
		callback=mueller_video.cb_slider,
	)

dpg.create_viewport(title='Spectro-polarimetric Visualizer')
dpg.set_primary_window('main_window', True)
dpg.setup_dearpygui()
dpg.show_viewport()

common_state.recent_files = io_utils.get_recent_files(30)
io_utils.update_history_checkboxes()

dpg.start_dearpygui()
dpg.destroy_context()