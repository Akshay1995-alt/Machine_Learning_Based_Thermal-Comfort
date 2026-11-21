# ASHRAE Thermal Comfort Machine Learning

This repository contains two sequential projects using the ASHRAE Global
Thermal Comfort Database II. Project 1 is the current focus; Project 2 starts
only after Project 1 is complete and begins again from the original CSV files.

## Project 1: Thermal sensation

The first project predicts the numeric `thermal_sensation` vote from
environmental and building-context features. The database includes fractional
votes, so the primary task preserves the recorded values and uses regression.
The analysis is global rather than limited to one country.

Follow the beginner-friendly workflow in [DAYWISE_PLAN.md](DAYWISE_PLAN.md)
and work through the notebooks in [notebooks/README.md](notebooks/README.md).

## Data

Place the two original database CSV files in `data/raw/` as described in the
notebooks. Do not modify the source files or commit the dataset; it is not
included in this repository update.

## Setup

From the repository root, install the packages with:

```powershell
python -m pip install -r requirements.txt
```

Open the notebooks in VS Code or Jupyter using the same Python environment.
The notebooks expect the project data and generated files to remain in the
repository's `data/` folders.

## Important limitations

The database contains observational measurements and occupant votes. Model
predictions and feature explanations describe associations; they do not prove
that changing a setpoint or fan speed causes a comfort outcome. These
notebooks do not measure HVAC energy use and should not be used to claim
measured energy savings.
