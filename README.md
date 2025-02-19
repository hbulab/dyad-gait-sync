# Gait Synchronisation in the Wild

This project contains the code for the analysis of gait synchronisation in dyads.

## Installation

To install the required packages, run the following command in the terminal:

```bash
pip install -r requirements.txt
```

The `pedestrians_social_binding` is installed from the source code in the [ATC - DIAMOR - Pedestrians](https://github.com/hbulab/atc-diamor) repository. It will be cloned and installed in the `pedestrians_social_binding` folder.
Refer to the README file in the `pedestrians_social_binding` folder for more information on how to get and prepare the data.

The pywct package is used to compute the wavelet coherence transform. Some modifications were made to the original package to be compatible with recent versions of the numpy package. In particular, `np.int` should be replaced by `int`.
Additionally, the `pywct` package was modified to prevent an error to be raised when the stationarity of the signal is checked. In particular, in file `wavelet.py` line 506, `a1, b1, c1 = ar1(y1)` and `a2, b2, c2 = ar1(y2)` should only be computed if `sig` is `True`.

## Usage

You can run the python scripts in the `src` folder to reproduce the results of the article. Some scripts compute some intermediate results (stored in pickle format in `data/pickle`). Results are stored in csv files in the `data/csv` folder, figures are stored in the `figures` folder and LaTeX tables are stored in the `tables` folder.
