import julia
import os


def __setup():
    j = julia.Julia()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    watershed_path = current_dir + '/watershed/watershed_fn.jl'

    # Get access to the watershed function
    watershed = j.eval('include("' + watershed_path + '")')

    return watershed


watershed = __setup()
