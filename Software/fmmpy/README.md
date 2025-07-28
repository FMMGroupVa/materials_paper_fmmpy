# fmmpy

**fmmpy** is a Python package for Frequency-Modulated Möbius (FMM) signal decomposition.  
It provides efficient tools for multichannel signal modeling, constrained parameter estimation, and component analysis.

This package implements the core methodology described in the paper:

> *PyFMM: A Python module for Frequency-Modulated Möbius Signal Decomposition*  
> Christian Canedo, Rocío Carratalá-Sáez, Cristina Rueda  
> [Submitted, 2025]  
> Repository: [ModulePyFMM on GitHub](https://github.com/FMMGroupVa/ModulePyFMM)

---

## Features

- Modular implementation of the FMM model
- Support for constrained fitting (phase, frequency, and shape parameters)
- AFD-based initialization and efficient backfitting algorithm
- Multichannel signal decomposition
- Confidence intervals for FMM parameters
- Visualization tools (components, predictions, residuals)

---

## Installation

```bash
pip install fmmpy