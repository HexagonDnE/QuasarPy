import pandas as pd
import subprocess
import tempfile
import os
import pathlib


"""
This module enables the use of Quasar functionalities from Python.

Every attribute, data or parameter used will be declared as Python data types, either as pandas DataFrames or numpy
array. In order for these files/data to be considered by Quasar, they will be temporarily converted into Quasar
supported data (.csv or .qsr files). The execution of this Quasar script will be carried out by "QuasarEngine_2.exe"
and, again, will output a Quasar-supported data file that will betransformed, saved, and returned to Python as a .csv.
"""

# use environmental variable to get Quasar Path
if "ODYSSEE_CAE_INSTALLDIR" in os.environ:
    _quasar_path = pathlib.Path(os.environ["ODYSSEE_CAE_INSTALLDIR"]).joinpath("CAE").with_name("QuasarEngine_2.exe")
else:
    _quasar_path = None
    print("ODYSSEE_CAE_INSTALLDIR not in environment, QuasarEngine_2.exe not located.")
    print("Set QuasarEngine_2.exe path with set_quasar_path")

_last_subprocess_out = None


def set_quasar_exec(quasar_exec):
    global _quasar_path
    _quasar_path = quasar_exec


def get_quasar_exec():
    return _quasar_path


def get_quasar_out():
    if _last_subprocess_out is not None:
        return _last_subprocess_out.stdout.decode().strip()
    else:
        return None


def _run_quasar(input_data, output_keys, quasar_script):
    """
    Run Quasar script

    Parameters
    ----------
    input_data: dict of {str: numpy.ndarray}.
        Contains input data matrices to be temporarily transformed to csv readable by Quasar.
    output_keys: list of str.
        Contains the name of the output files.
    quasar_script: str.
        Quasar script to be executed containing the corresponding parameters according to the function to be executed.

    Return
    ------
    out: dict of {str: numpy.ndarrya}
        Contain the output variables that, in most cases, will be a numpy.ndarray.
    None: *if errors are found.
    """

    global _last_subprocess_out
    with tempfile.TemporaryDirectory() as tmpdirname:

        # write input data to files
        for key, value in input_data.items():
            input_file_name = os.path.join(tmpdirname, key + ".csv")
            pd.DataFrame(value).to_csv(input_file_name, header=None, index=None, sep=";")

        # create common quasar script preamble
        quasar_script_preamble = f"""import "QuasarMatrix"\nimport "QuasarExternal"\nString cwd = "{tmpdirname}"\n"""

        # write quasar script
        quasar_script_name = os.path.join(tmpdirname, "q.qsr")
        with open(quasar_script_name, "w") as f:
            f.write(quasar_script_preamble)
            f.write(quasar_script)

        # execute quasar script
        _last_subprocess_out = subprocess.run([_quasar_path, quasar_script_name], capture_output=True, cwd=tmpdirname)

        # if there was no printed output from quasar script, assume success
        if _last_subprocess_out.stdout.decode() == "":  # Nothing printed to console during Quasar Execution
            out = {}
            for key in output_keys:
                output_file_name = os.path.join(tmpdirname, key + ".csv")
                value = pd.read_csv(output_file_name, sep=";", header=None).values
                out[key] = value

            if len(output_keys) == 1:
                return out[output_keys[0]]
            else:
                return out

        else:
            return None


if __name__ == "__main__":
    import numpy as np

    # quasar path pulled from environment
    print("Quasar Path:", get_quasar_exec())

    # test _run_quasar with identity pass
    print("Test Identity Pass")
    X = np.random.random((5, 5))
    input_data = {"X": X}
    output_keys = ["Y"]
    quasar_script = """Matrix X = loadCsv(cwd+"\X.csv")\nX.saveCsv(cwd+"\Y.csv", "%16.8E")"""  # noqa
    Y = _run_quasar(input_data, output_keys, quasar_script)
    print("L2 Norm of Identity Pass:", np.linalg.norm(X - Y))
    print("Last Quasar Out:", get_quasar_out())

    # test _run_quasar script that will result in a printed output (usually an error)
    print("Test Printed Output")
    print(_run_quasar({}, [], 'print("Quasar Out")'))
    print("Last Quasar Out:", get_quasar_out())
