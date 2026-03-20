corrections = ["Original", "Gamma", "m00", "m00 (Keep Intensity)"]
mueller_rgb_options = ["Positive", "Negative"]
rgb_map = {"R": 0, "G": 1, "B": 2}
mueller_selected_channel = "R"
mueller_selected_correction = "Original"
gamma = 2.2

mueller_visualizing = "Original" # ["Original", "Gamma", "m00", "Positive", "Negative"]
visualizing_gamma = False

is_video = False

# --- Lu-Chipman View State ---
visualization_mode = "Matrix"  # "Matrix" or "Decomposition"
decomposition_options = [
    "Depolarization Index",
    "Diattenuation",
    "Polarizance",
    "Linear Retardance",
    "Matrix: Diattenuator",
    "Matrix: Retarder",
    "Matrix: Depolarizer",
    "Matrix: Corrected (M_Delta*M_R)"
]
selected_decomposition = "Depolarization Index"