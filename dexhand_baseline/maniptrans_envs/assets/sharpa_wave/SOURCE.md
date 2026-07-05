# Sharpa Wave Assets

Source: https://github.com/sharpa-robotics/sharpa-urdf-usd-xml

Commit: 6eea427eb24189519f32b9f21674cd534d3f973c

License: Apache-2.0, copied in `LICENSE.txt`.

Imported files:

- `wave_01/right_sharpa_wave/right_sharpa_wave.urdf` -> `sharpa_wave_right.urdf`
- `wave_01/left_sharpa_wave/left_sharpa_wave.urdf` -> `sharpa_wave_left.urdf`
- `wave_01/right_sharpa_wave/meshes/*.STL`
- `wave_01/left_sharpa_wave/meshes/*.STL`

Local changes:

- Replaced ROS `package://*_sharpa_wave/meshes/` mesh URIs with `./meshes/`
  so IsaacGym can load the assets from this directory directly.
