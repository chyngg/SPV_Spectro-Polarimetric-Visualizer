from . import state
import dearpygui.dearpygui as dpg
from .visualization import update_visualization, update_wavelengths_visualization
from .histogram import show_stokes_histogram

def change_direction(sender, app_data):
	state.selected_direction = app_data
	show_stokes_histogram(parameter=state.shown_histogram, direction=state.selected_direction)

def show_histogram_callback(parameter):
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
	dpg.configure_item("polarimetric_options", enabled=True)
	dpg.configure_item("wavelength_options", enabled=True)
	update_wavelengths_visualization(state.selected_wavelength, state.selected_stokes)

def reload_visualization():
	(state.vmin, state.vmax) = (state.input_vmin, state.input_vmax)
	if not state.visualizing_by_wavelength:
		update_visualization(state.selected_option)
	else:
		update_wavelengths_visualization(state.selected_wavelength, state.selected_option)

def reset_visualization():
	if state.selected_option == "s0" or state.selected_option == "dolp" or state.selected_option == "docp":
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
		state.lower_right_y = float(dpg.get_value("upper_right_y"))
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

