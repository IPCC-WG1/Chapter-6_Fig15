
# Code to produce Fig 6.15 in the IPCC WG1 6th assessment report
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7957964.svg)](https://doi.org/10.5281/zenodo.7957964)

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

For documentation, see the notebook [delta_T_CO2_plots_final.ipynb](delta_T_CO2_plots_final.ipynb) or [documentation/delta_T_CO2_plots_final.pdf](documentation/delta_T_CO2_plots_final.pdf).

## Input data: 
Input data is in [input_data/recommended_irf_from_2xCO2_2021_02_25_222758.csv](input_data/recommended_irf_from_2xCO2_2021_02_25_222758.csv) which contains the coefficients for the recommended IRF used in AR6. 
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
- Operating System: Linux ( Ubuntu 20.04)
- Environment required to compile and run: See [requirements.txt](requirements.txt) or run
```bash
conda env create -f environment.yml
conda activate ipcc_fig6_15
```
in the base directory. 

## License: 
Apache 2.0

## How to cite:
When citing this code, please include both the code citation and the following citation for the related report component:

Figure 6.15 in IPCC, 2021: Chapter 6. In: Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Szopa, S., V. Naik, B. Adhikary, P. Artaxo, T. Berntsen, W.D. Collins, S. Fuzzi, L. Gallardo, A. Kiendler-Scharr, Z. Klimont, H. Liao, N. Unger, and P. Zanis, 2021: Short-Lived Climate Forcers. In Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Masson-Delmotte, V., P. Zhai, A. Pirani, S.L. Connors, C. Péan, S. Berger, N. Caud, Y. Chen, L. Goldfarb, M.I. Gomis, M. Huang, K. Leitzell, E. Lonnoy, J.B.R. Matthews, T.K. Maycock, T. Waterfield, O. Yelekçi, R. Yu, and B. Zhou (eds.)]. Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA, pp. 817–922, doi: 10.1017/9781009157896.008 .]
