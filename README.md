# Dry and moist convective upper bounds for near-surface temperatures - code

## Overview

This folder contains code to reproduce the figures in Nicolas & Hotz (2026): Dry and moist convective upper bounds for near-surface temperatures.  

## Requirements

A .yml file is included that contains all necessary python packages to run the code. You can create the conda environment using `conda env create -f environment.yml`, then activate with `conda activate atmos`. See [this link](https://www.anaconda.com/docs/getting-started/miniconda/main) for more info on how to install conda.

## Reproducing Figures

The Jupyter notebook `make_figures.ipynb` provides instructions and code to produce the Figures. 

It uses various tools from `general_tools.py`, in particular for binning data, dealing with station data, performing vertical interpolations or calculating parcel profiles, and a plotting utility from `plotting_tools.py`

## Data availability

The code relies on data from ERA5 (available [here](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=overview)) and IGRA2 (available through the [igra python package](https://pypi.org/project/igra/)).

For convenience, we provide processed data from both sources in a dataset hosted on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20125256.svg)](https://doi.org/10.5281/zenodo.20125256). Simply download the dataset, change the DATA_PATH variable in the preamble of `make_figures.ipynb` to the dataset's location, and you'll be able to reproduce the figures.

## Citation

If you use any of the code in your work, we simply ask you to cite the paper. 

The code itself is also citable through Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20125545.svg)](https://doi.org/10.5281/zenodo.20125545)

For any questions, do not hesitate to email me!
