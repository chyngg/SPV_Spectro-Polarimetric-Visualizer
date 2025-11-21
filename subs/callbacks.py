from subs.SP_image import sp_state
from subs.Mueller_matrix_image import mueller_state, mueller_video
from subs import common_state
import dearpygui.dearpygui as dpg
from subs.SP_image.sp_visualization import update_visualization, update_wavelengths_visualization
from subs.Mueller_matrix_image.mueller_visualization import visualize_rgb_mueller_grid, visualize_rgb_mueller_rgbgrid
from subs.SP_image.histogram import show_stokes_histogram

def change_direction(sender, app_data):
	sp_state.selected_direction = app_data
	show_stokes_histogram(parameter=sp_state.shown_histogram, direction=sp_state.selected_direction)

def show_histogram_callback(parameter):
	if common_state.current_tab == "RGB_Mueller":
		return
	show_stokes_histogram(parameter=parameter, direction=sp_state.selected_direction)

def set_sp_option(sender):
	if common_state.current_tab == "Trichromatic" and sender == "original":
		sp_state.sp_visualizing = "original"
	elif common_state.current_tab == "Hyperspectral" and sender == "original":
		sp_state.sp_visualizing = "original_hyper"
	else:
		sp_state.sp_visualizing = sender
	update_visualization(sp_state.sp_visualizing)

def set_sp_by_channel_callback():
	sp_state.selected_wavelength = dpg.get_value("wavelength_options")
	sp_state.selected_stokes = dpg.get_value("polarimetric_options")
	update_wavelengths_visualization(sp_state.selected_wavelength, sp_state.selected_stokes)

def change_polarized_option_callback(sender):
	sp_state.now_polarized = dpg.get_value("select_polarized_option")
	if sp_state.sp_visualizing in ["Polarized(Linear)", "Polarized(Circular)", "Polarized(total)"]:
		sp_state.sp_visualizing = f"Polarized({sp_state.now_polarized})"
		update_visualization(sp_state.sp_visualizing)

def activate_visualization():
	if common_state.current_tab == "RGB_Mueller":
		return
	dpg.configure_item("polarimetric_options", enabled=True)
	dpg.configure_item("wavelength_options", enabled=True)
	update_wavelengths_visualization(sp_state.selected_wavelength, sp_state.selected_stokes)

def reload_visualization():
	(common_state.vmin, common_state.vmax) = (common_state.input_vmin, common_state.input_vmax)
	if sp_state.visualizing_by_wavelength:
		update_wavelengths_visualization(sp_state.selected_wavelength, common_state.selected_option)
	elif common_state.current_tab != "RGB_Mueller" and common_state.current_tab != "Mueller_video":
		update_visualization(common_state.selected_option)
	elif common_state.current_tab == "RGB_Mueller": #RGB_Mueller
		if common_state.vmin < common_state.vmax:
			current_channel = common_state.wavelength_options[mueller_state.mueller_selected_channel]
			visualize_rgb_mueller_grid(common_state.npy_data, channel=current_channel, vmin=common_state.vmin, vmax=common_state.vmax)
	else: # Mueller_video
		mueller_video.on_mode_or_channel_changed()

def reset_visualization():
	if common_state.current_tab == "RGB_Mueller":
		current_channel = common_state.wavelength_options[mueller_state.mueller_selected_channel]
		(common_state.vmin, common_state.vmax) = (-1, 1)
		visualize_rgb_mueller_grid(common_state.npy_data, channel=current_channel, vmin=-1, vmax=1)
	else:
		if common_state.selected_option == "s0" or common_state.selected_option == "DoLP" or common_state.selected_option == "DoCP":
			(common_state.vmin, common_state.vmax) = (0, 1)
		else:
			(common_state.vmin, common_state.vmax) = (-common_state.temp_abs_vmax, common_state.temp_abs_vmax)
		update_visualization(common_state.selected_option)

def crop_graph_option_callback():
	sp_state.crop_graph_option = dpg.get_value("crop_graph_options")

def multi_graph_option_callback():
	sp_state.multi_graph_option = dpg.get_value("multi_graph_options")

def on_close_graph_window():
	sp_state.show_rectangle_overlay = False
	update_visualization(common_state.selected_option)

def on_lower_left_x():
	try:
		sp_state.lower_left_x = float(dpg.get_value("lower_left_x"))
	except ValueError:
		return

def on_lower_left_y():
	try:
		sp_state.lower_left_y = float(dpg.get_value("lower_left_y"))
	except ValueError:
		return

def on_upper_right_x():
	try:
		sp_state.upper_right_x = float(dpg.get_value("upper_right_x"))
	except ValueError:
		return

def on_upper_right_y():
	try:
		sp_state.upper_right_y = float(dpg.get_value("upper_right_y"))
	except ValueError:
		return

def on_vmax_change():
	try:
		vmax_ = float(dpg.get_value("vmax_input"))
		common_state.input_vmax = vmax_
	except ValueError:
		common_state.vmax = 0

def on_vmin_change():
	try:
		vmin_ = float(dpg.get_value("vmin_input"))
		common_state.input_vmin = vmin_
	except ValueError:
		common_state.vmin = 0


# ---- For mueller-matrix visualization ----

def mueller_select_option_callback():
    mueller_state.mueller_selected_correction = dpg.get_value("mueller_correction")

    # 스칼라 모드일 때는 correction 선택과 mueller_visualizing을 동기화
    if mueller_state.mueller_visualizing in ["Original", "m00", "Gamma"]:
        mueller_state.mueller_visualizing = mueller_state.mueller_selected_correction

    # 비디오면: 현재 프레임만 새 설정으로 다시 그림
    if mueller_state.is_video:
        mueller_video.on_mode_or_channel_changed()
        return

    # 단일 프레임 RGB Mueller (기존 동작)
    if mueller_state.mueller_visualizing in ["Original", "m00", "Gamma"]:
        visualize_rgb_mueller_grid(
            common_state.npy_data,
            channel=mueller_state.mueller_selected_channel,
            correction=mueller_state.mueller_selected_correction,
            vmin=-1,
            vmax=1,
        )
    else:
        visualize_rgb_mueller_rgbgrid(
            common_state.npy_data,
            mueller_state.mueller_selected_correction,
            sign=mueller_state.mueller_visualizing,
        )

def mueller_channel_callback():
    mueller_state.mueller_selected_channel = dpg.get_value("mueller_channel")
    mueller_state.mueller_visualizing = mueller_state.mueller_selected_correction

    if mueller_state.is_video:
        mueller_video.on_mode_or_channel_changed()
    else:
        mueller_select_option_callback()

def mueller_rgb_callback_positive():
    mueller_state.mueller_visualizing = "Positive"
    if mueller_state.is_video:
        mueller_video.on_mode_or_channel_changed()
    else:
        visualize_rgb_mueller_rgbgrid(
            common_state.npy_data,
            mueller_state.mueller_selected_correction,
            sign="Positive",
        )

def mueller_rgb_callback_negative():
    mueller_state.mueller_visualizing = "Negative"
    if mueller_state.is_video:
        mueller_video.on_mode_or_channel_changed()
    else:
        visualize_rgb_mueller_rgbgrid(
            common_state.npy_data,
            mueller_state.mueller_selected_correction,
            sign="Negative",
        )

def on_gamma_change():
    mueller_state.gamma = float(dpg.get_value("gamma_input"))
    if not mueller_state.visualizing_gamma:
        return

    if mueller_state.is_video:
        mueller_video.on_mode_or_channel_changed()
    else:
        if mueller_state.mueller_visualizing in mueller_state.mueller_rgb_options:
            visualize_rgb_mueller_rgbgrid(
                common_state.npy_data,
                mueller_state.mueller_selected_correction,
                sign=mueller_state.mueller_visualizing,
            )
        else:
            visualize_rgb_mueller_grid(
                common_state.npy_data,
                channel=mueller_state.mueller_selected_channel,
                correction=mueller_state.mueller_selected_correction,
                vmin=-1,
                vmax=1,
            )
