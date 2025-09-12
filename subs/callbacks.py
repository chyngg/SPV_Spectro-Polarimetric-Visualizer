from . import state
import dearpygui.dearpygui as dpg
from .visualization import update_visualization, update_wavelengths_visualization, visualize_rgb_mueller_grid
from .histogram import show_stokes_histogram

def change_direction(sender, app_data):
	state.selected_direction = app_data
	show_stokes_histogram(parameter=state.shown_histogram, direction=state.selected_direction)

def show_histogram_callback(parameter):
	if state.current_tab == "RGB_Mueller":
		return
	show_stokes_histogram(parameter=parameter, direction=state.selected_direction)

def set_selected_option(sender):
	if state.current_tab == "Trichromatic" and sender == "original":
		update_visualization("original")
	elif state.current_tab == "Hyperspectral" and sender == "original":
		update_visualization("original_hyper")
	else:
		update_visualization(sender)

def select_option_callback():
	state.selected_wavelength = dpg.get_value("wavelength_options")
	state.selected_stokes = dpg.get_value("polarimetric_options")
	update_wavelengths_visualization(state.selected_wavelength, state.selected_stokes)

def activate_visualization():
	if state.current_tab == "RGB_Mueller":
		return
	dpg.configure_item("polarimetric_options", enabled=True)
	dpg.configure_item("wavelength_options", enabled=True)
	update_wavelengths_visualization(state.selected_wavelength, state.selected_stokes)

def reload_visualization():
	(state.vmin, state.vmax) = (state.input_vmin, state.input_vmax)
	if state.visualizing_by_wavelength:
		update_wavelengths_visualization(state.selected_wavelength, state.selected_option)
	elif state.current_tab != "RGB_Mueller":
		update_visualization(state.selected_option)
	else: #RGB_Mueller
		if state.vmin < state.vmax:
			current_channel = state.rgb_map[state.mueller_selected_channel]
			visualize_rgb_mueller_grid(state.npy_data, channel=current_channel, vmin=state.vmin, vmax=state.vmax)

def reset_visualization():
	if state.current_tab == "RGB_Mueller":
		current_channel = state.rgb_map[state.mueller_selected_channel]
		(state.vmin, state.vmax) = (-1, 1)
		visualize_rgb_mueller_grid(state.npy_data, channel=current_channel, vmin=-1, vmax=1)
	else:
		if state.selected_option == "s0" or state.selected_option == "DoLP" or state.selected_option == "DoCP":
			(state.vmin, state.vmax) = (0, 1)
		else:
			(state.vmin, state.vmax) = (-state.temp_abs_vmax, state.temp_abs_vmax)
		update_visualization(state.selected_option)

def crop_graph_option_callback():
	state.crop_graph_option = dpg.get_value("crop_graph_options")

def multi_graph_option_callback():
	state.multi_graph_option = dpg.get_value("multi_graph_options")

def on_close_graph_window():
	state.show_rectangle_overlay = False
	update_visualization(state.selected_option)

def on_lower_left_x():
	try:
		state.lower_left_x = float(dpg.get_value("lower_left_x"))
	except ValueError:
		return

def on_lower_left_y():
	try:
		state.lower_left_y = float(dpg.get_value("lower_left_y"))
	except ValueError:
		return

def on_upper_right_x():
	try:
		state.upper_right_x = float(dpg.get_value("upper_right_x"))
	except ValueError:
		return

def on_upper_right_y():
	try:
		state.upper_right_y = float(dpg.get_value("upper_right_y"))
	except ValueError:
		return

def on_vmax_change():
	try:
		vmax_ = float(dpg.get_value("vmax_input"))
		state.input_vmax = vmax_
	except ValueError:
		state.vmax = 0

def on_vmin_change():
	try:
		vmin_ = float(dpg.get_value("vmin_input"))
		state.input_vmin = vmin_
	except ValueError:
		state.vmin = 0

def mueller_select_option_callback():
	state.mueller_selected_channel = dpg.get_value("mueller_channel")
	state.mueller_selected_correction = dpg.get_value("mueller_correction")
	visualize_rgb_mueller_grid(state.npy_data, channel=state.rgb_map[state.mueller_selected_channel], vmin=-1, vmax=1)

