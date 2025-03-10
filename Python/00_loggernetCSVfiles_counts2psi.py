#!/usr/bin/env python
# coding: utf-8

# # Convert Phase 2 well data (July 21st 2022 onwards)
# Locher Environmental recorded data from multiple transducers in two adjacent wells near SLC 39A at Kennedy Space Center (KSC), between March and November, 2022. This was part of a seismo-acoustic erosion pilot experiment. During phase 2 of the experiment, which began on July 21st, 2022, vibrating wire sensors were used and found to give more accurate water levels. These data were captured on Campbell Scientific dataloggers and recorded in TOB3 binary file format. The purpose of this notebook is to:
# - read these files. initially this was done wit a Campbell Scientific python library, but for cross-check I converted them with LoggerNet
#   so the TOB3 binary file to CSV conversion is omitted here
# - apply calibration equations
# - write to 'corrected CSV' files
# Calibration constants are defined in the transducers dataframe, based on copied from the file "A Pz Linear Gage Calc_NASA Sensors.xls". 
# Other metadata included in this dataframe come from sections 6 and 7 in the file "2022-12-03_ Field Sheet for Deployment of Groundwater Equipment at NASA_Part_II.pdf".
# 
# Note on *TIME ZONES*: 
# - Local time is used in Campbell Scientific binary files, and files converted to Pickle.

# raw data on April 1, 2022 from 16:10 to 16:40 UTC. Launch from SLC40 at 16:24 UTC, watched from Titusville
import header
paths = header.setup_environment()
import Python.libWellData_wrong as LLE

# step to generate a list of TOB3 files and convert them with Python library or LoggerNet is omitted here

# Generate complete list of LoggerNet CSV files (converted from TOB3 files)
csvfiles = LLE.list_loggernet_csv_files(paths['TOB3_DIR'])
print(len(csvfiles))
LLE.correct_csvfiles(csvfiles, paths, converted_by_LoggerNet=True, MAXFILES=None, keep_existing=True) # conversion equations are applied here

