
# CSI-Thesis-2025
Contains code relative to Master Thesis completed at CSI-Nanolab in 2025,notably the 3 pipelines developped and used to obtain the main result and some test data.

2 requirements.txt files were included:
- GP_requirements.txt (for virtual python environment)
- GP_curv_requirements.txt (for anaconda environment)

1. Polarization analysis
- Script file: pipeline_polarization_spectrum_sine_fit.ipynb
- Test data included: 2D Confocal, 3D LLSM and 2D confocal GPMVs datasets
- Needs to be run with the GP_requirements

2. Phase separated GPMVs analysis
- Includes multiple main script files:
    - Napari_GP_pipeline.py -> main pipeline, best used for individual file analysis
    - distribution_parser.py -> copy for main pipeline, processes series of files
    - distribution_analysis.ipynb -> processes the csv files generated with distribution_parser.py
- Test data included: 3D LLSM GPMVs dataset
- Needs to be run with the GP_requirements and in the same folder as the GP_modules file

3. GP-curvature correlation analysis
- Script file: updated_curv_analysis.ipynb
- Test data included: GP layers for cell and GPMV 3D LLSM datasets
- Test results included: Filtered, curvature mapped and GP mapped meshes for cell and GPMV. Also includes respective raw curvature and GP data.
- Needs to be run with the GP_requirements and in the same folder as the unwrap3D file
