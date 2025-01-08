#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:22:23 2024

@author: mike
"""
import pathlib
from core import File







######################################################
### Parameters

base_path = pathlib.Path('/media/nvme1/data/ecmwf/era5-land')
file = base_path.joinpath('test1.nc')



#####################################################3



def cf_checks(file, **kwargs):
    """

    """
    f = File(file, **kwargs)





def check_coordinates(f):
    """

    """


from compliance_checker.runner import ComplianceChecker, CheckSuite

# Load all available checker classes
check_suite = CheckSuite()
check_suite.load_all_available_checkers()

# Run cf and adcc checks
path = str(base_path.joinpath('2m_temperature_1950-1957_reanalysis-era5-land.nc'))
checker_names = ['cf']
verbose = 0
criteria = 'normal'
output_filename = str(base_path.joinpath('cf_check.json'))
output_format = 'json'
"""
Inputs to ComplianceChecker.run_checker

path            Dataset location (url or file)
checker_names   List of string names to run, should match keys of checkers dict (empty list means run all)
verbose         Verbosity of the output (0, 1, 2)
criteria        Determines failure (lenient, normal, strict)
output_filename Path to the file for output
output_format   Format of the output

@returns                If the tests failed (based on the criteria)
"""
return_value, errors = ComplianceChecker.run_checker(path,
                                                     checker_names,
                                                     verbose,
                                                     criteria,
                                                     output_filename=output_filename,
                                                     output_format=output_format)

# Open the JSON output and get the compliance scores
with open(output_filename, 'r') as fp:
    cc_data = json.load(fp)
    scored = cc_data[cc_test[0]]['scored_points']
    possible = cc_data[cc_test[0]]['possible_points']
    log.debug('CC Scored {} out of {} possible points'.format(scored, possible))




























































