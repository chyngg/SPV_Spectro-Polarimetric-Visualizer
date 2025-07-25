# SPV: Spectro-Polarimetric Visualizer


[[Spectral-Polarimetric Data]](https://huggingface.co/datasets/jyj7913/spectro-polarimetric)

<p align="center">
  <img src="/assets/readme/RGB_1.png" alt="RGB Image" width="45%" />
  <img src="/assets/readme/Hyperspectral_1.png" alt="Hyperspectral Image" width="45%" />
</p>

## Installation & Running the code

```bash
git clone https://github.com/chyngg/SPV_Spectro-Polarimetric-Visualizer.git
cd SPV_Spectro-Polarimetric-Visualizer
pip install dearpygui dearpygui-extend matplotlib numpy opencv-python scipy
python main.py
```

## Features


- Visualize for entire wavelengths:
    - Full Stokes parameters (`s0`, `s1`, `s2` , `s3`)
    - Polarization feature maps (`DoLP`, `AoLP`, `DoCP`, `CoP`)
    - Unpolarized/Polarized light
    - Histogram visualization
        - Stokes vector distribution
        - Gradient distributions
        - Gradient derivatives distribution
        - Polarization feature gradients (`DoLP`, `AoLP`, `DoCP`, `CoP`)
- Visualize for individual wavelengths:
    - Full Stokes parameters (`s0`, `s1`, `s2` , `s3`)
    - Polarization feature maps (`DoLP`, `AoLP`, `DoCP`, `CoP`)
- RGB approximation from hyperspectral data
- Region-based graph plotting accross wavelengths
    - Stokes parameters (`s0`, `s1`, `s2` , `s3`)
    - Polarization features (`DoLP`, `AoLP`, `DoCP`, `CoP`)
- Save the visualization as `.png`