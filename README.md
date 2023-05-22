
# Code to produce Fig 6.15 in the IPCC WG1 6th assessment report

This is Python  code to produce IPCC AR6 WGI Figure 6.15<br>
Creator: Sara Blichner, University of Oslo (now Stockholm University)<br>
Contact: sara.blichner@gmail.com or sara.blichner@aces.su.se<br>
Last updated on: Feb 23, 2022


## Code functionality: 
To make figure, run:
```bash
python delta_T_CO2_plots_final.py
```
Plots are saved in the repository "plots/"

For documentation, see the notebook [delta_T_CO2_plots_final.ipynb](delta_T_CO2_plots_final.ipynb) or [delta_T_CO2_plots_final.pdf](delta_T_CO2_plots_final.pdf).

## Input data: 
Input data is in [recommended_irf_from_2xCO2_2021_02_25_222758.csv](recommended_irf_from_2xCO2_2021_02_25_222758.csv) which contains the coefficients for the recommended IRF used in AR6. 
## Output variables:

The code plots the figure as in the report.

## Information on  the software used
 - Software Version: Python 3.7. See also [requirements.txt](requirements.txt)
 - Landing page to access the software: https://www.python.org/downloads/ or first download and install https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html 
and then in  the base repository run: 
```bash
conda env create -f environment.yml
conda activate ipcc_fig6_15
```
- Operating System: Ubuntu 
- Environment required to compile and run: See [requirements.txt](requirements.txt).

## License: 
Apache 2.0

## How to cite:
Sara Blichner (2021) "IPCC-WG1/Chapter-6_Fig15", https://github.com/IPCC-WG1/Chapter-6_Fig15 
