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
        return _last_subprocess_out.stdout.decode()
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


def _direct_interpolator(
    x,
    y,
    xq,
    arg_num,
    Neighbours,
    Power,
    BasisFunction,
    StationarityFunction,
    NuggetEffect,
    Pulsation,
    InterpMethod,
    Radius,
    Normalize,
):
    """
    Function to be called by any direct interpolator method.

    Each of the direct interpolators calling this function will provide the value of the arguments to be used and set
    the rest to 0.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    arg_num: int {1: ARBF or RBF, 2: kriging, 5: INVD}
        Specify kind of direct interpolation to be performed.
    Neighbours: int
        Number of neighbours to reference.
    Power: float, {0, 0.25, 0.5, 2.0, 4.0}
        Order of interpoaltor function.
    BasisFunction: int, {0, 1, 2, 3, 4, 5}
        Define basis function (type of interpolation).
        0: 'none'.
        1: 'constant'.
        2: 'linear'.
        3: 'quadratic'.
        4: 'cubic'.
        5: 'trigonometric'.
    StationarityFunction: int, {0, 1, 2, 3, 4}
        Define stationarity function (capacity of a response function to change its behavior).
        0: 'h1'.
        1: 'h2'.
        2: 'h3'.
        3: 'exp'.
        4: 'matern32'.
    NuggetEffect: float
        Real values used for trigonometic radial basis function.
        0.0: not used.
    Pulsation: float
        Real values used for trigonometic radial basis function.
        0.0: not used.
    InterpMethod: int, {0, 1, 2, 3, 4}
        Define a radial basis function (type of interpolation)
        0: 'multiquadratic'.
        1: 'invMultiquadratic'.
        2: 'thinPlateSpline'.
        3: 'gaussian'.
        4. 'linear'.
    Radius: float
        0.0: to be automatically calculated from data.
    Normalize: int, {0, 1, 2}
        Type of normalization to be performed
        0: No normalization
        1: Min/Max normalization.
        2: Variance normalization.

    Returns
    -------
    yq: numpy.ndarray
        Output predicitons corresponding to ``xq``.
    """
    input_data = {"X": x, "Y": y, "XQ": xq}

    output_keys = ["YQ"]

    # fmt: off
    quasar_script = f"""
Matrix train_X=loadCsv(cwd+"\X.csv")
Matrix train_Y=loadCsv(cwd+"\Y.csv")
Matrix query_X=loadCsv(cwd+"\XQ.csv")
int arg_num={arg_num}
    """ # noqa
    # fmt: on

    if arg_num == 2:  # Kriging
        # fmt: off
        quasar_script += f"""
        int BasisFunction={BasisFunction}
        int StationarityFunction={StationarityFunction}
        double NuggetEffect={NuggetEffect}
        double Pulsation={Pulsation}
        Matrix YN
        YN=ext("interpolator", "matrix", "forecast", train_X, train_Y, query_X, arg_num, BasisFunction, StationarityFunction, 1, NuggetEffect, Pulsation)
        YN.saveCsv(cwd+"\YQ.csv","%16.8E")
        """ # noqa
        # fmt: on
    elif arg_num == 5:  # Inverse Distance
        # fmt: off
        quasar_script += f"""
        int Neighbours={Neighbours}
        double Power={Power}
        Matrix YN
        YN=ext("interpolator", "matrix", "forecast", train_X, train_Y, query_X, arg_num, Neighbours, Power)
        YN.saveCsv(cwd+"\YQ.csv","%16.8E")
        """ # noqa
        # fmt: on
    elif arg_num == 1:  # Radius Basis Function
        # fmt: off
        quasar_script += f"""
        int InterpMethod={InterpMethod}
        int Normalize={Normalize}
        double Radius={Radius}
        Matrix YN
        YN=ext("interpolator", "matrix", "forecast", train_X, train_Y, query_X, arg_num, InterpMethod, Radius, Normalize)
        YN.saveCsv(cwd+"\YQ.csv","%16.8E")
        """ # noqa
        # fmt: on
    return _run_quasar(input_data, output_keys, quasar_script)


def kriging(x, y, xq, BasisFunction="linear", StationarityFunction="h1", NuggetEffect=None, Pulsation=None):
    """
    Kriging interpolator function.

    Called by the user providing necessary input data. This function will check the format of the input arguments and
    call ``QuasarPy._direct_interpolator``.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    BasisFunction: int or str, default 'linear'
        Define basis function (type of interpolation).

        0: 'none'.
        1: 'constant'.
        2: 'linear'.
        3: 'quadratic'.
        4: 'cubic'.
        5: 'trigonometric'.

    StationarityFunction: int or str, default 'h1'
        Define stationarity function (capacity of a response function to change its behavior).

        0: 'h1',
        1: 'h2',
        2: 'h3',
        3: 'exp',
        4: 'matern32'.

    NuggetEffect: float, default 0.0
        Real values used for trigonometic radial basis function.

        0.0: not used.

    Pulsation: float, default 0.0
        Real values used for trigonometic radial basis function.

        0.0: not used.

    Returns
    -------
    numpy.ndarray
        Output predicitons corresponding to ``xq``, extracted from ``QuasarPy._direct_interpolator``.
    """
    # Accept integer or string for BasisFunction
    if isinstance(BasisFunction, str):
        BasisFunctionMap = {"none": 0, "constant": 1, "linear": 2, "quadratic": 3, "cubic": 4, "trigonometric": 5}
        BasisFunctionInt = BasisFunctionMap[BasisFunction]
    elif isinstance(BasisFunction, int):
        assert 0 <= BasisFunction <= 5
        BasisFunctionInt = BasisFunction
    else:
        raise TypeError("BasisFunction must be int or str")

    # Accept integer or string for StationarityFunction
    if isinstance(StationarityFunction, str):
        StationarityFunctionMap = {"h1": 0, "h2": 1, "h3": 2, "exp": 3, "matern32": 4}
        StationarityFunctionInt = StationarityFunctionMap[StationarityFunction]
    elif isinstance(StationarityFunction, int):
        assert 0 <= StationarityFunction <= 4
        StationarityFunctionInt = StationarityFunction
    else:
        raise TypeError("StationarityFunction must be int or str")

    if NuggetEffect is None:
        NuggetEffect = 0.0

    if Pulsation is None:
        Pulsation = 0.0

    if BasisFunctionInt == 5:
        assert NuggetEffect > 0.0
        assert Pulsation > 0.0

    return _direct_interpolator(
        x,
        y,
        xq,
        arg_num=2,
        Neighbours=0,
        Power=0.0,
        BasisFunction=BasisFunctionInt,
        StationarityFunction=StationarityFunctionInt,
        NuggetEffect=NuggetEffect,
        Pulsation=Pulsation,
        InterpMethod=0,
        Radius=0.0,
        Normalize=0,
    )


def INVD(x, y, xq, Neighbours=3, Power=2.0):
    """
    Inverse distance interpolator function.

    Called by the user providing necessary input data. This function will check the format of the input arguments and
    call ``QuasarPy._direct_interpolator``.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    Neighbours: int, default 3
        Number of neighbours to reference.
    Power: float, {0, 0.25, 0.5, 2.0, 4.0}, default 2.0
        Order of interpoaltor function.

    Returns
    -------
    numpy.ndarray
        Output predicitons corresponding to ``xq``, extracted from ``QuasarPy._direct_interpolator``.
    """
    # Accept integer for Neighbours
    if isinstance(Neighbours, int):
        assert 0 <= Neighbours <= len(x)
    else:
        raise TypeError("Neighbours must be int")

    # Accept float for Power
    if isinstance(Power, float):
        assert Power in [0.0, 0.5, 0.25, 2.0, 4.0]
    else:
        raise TypeError("Power must be float")

    return _direct_interpolator(
        x,
        y,
        xq,
        arg_num=5,
        Neighbours=Neighbours,
        Power=Power,
        BasisFunction=0,
        StationarityFunction=0,
        NuggetEffect=0,
        Pulsation=0,
        InterpMethod=0,
        Radius=0.0,
        Normalize=0,
    )


def RBF(x, y, xq, InterpMethod="linear", Radius=0.0, Normalize=1):
    """
    RBF interpolator function.

    Called by the user providing necessary input data. This function will check the format of the input arguments and
    call ``QuasarPy._direct_interpolator``.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    InterpMethod: int or str, default 'linear'
        Define a radial basis function (type of interpolation)

        0: 'multiquadratic',
        1: 'invMultiquadratic',
        2: 'thinPlateSpline',
        3: 'gaussian',
        4. 'linear'.

    Radius: float, default 0.0

        0.0: to be automatically calculated from data.

    Normalize: int, default 1

        Type of normalization to be performed

        0: No normalization,
        1: Min/Max normalization,
        2: Variance normalization.

    Returns
    -------
    numpy.ndarray
        Output predicitons corresponding to ``xq``, extracted from ``QuasarPy._direct_interpolator``.
    """
    # Accept integer or string for BasisFunction
    if isinstance(InterpMethod, str):
        InterpMethodMap = {
            "multiquadric": 0,
            "invMultiquadric": 1,
            "thinPlateSpline": 2,
            "gaussian": 3,
            "linear": 4,
        }
        InterpMethodInt = InterpMethodMap[InterpMethod]
    elif isinstance(InterpMethod, int):
        assert 0 <= InterpMethod <= 4
        InterpMethodInt = InterpMethod
    else:
        raise TypeError("InterpMethod must be int or str")

    # Accept integer or string for Normalize
    if isinstance(Normalize, str):
        NormalizeMap = {"No": 0, "Min/Max": 1, "Variance": 2}
        NormalizeInt = NormalizeMap[Normalize]
    elif isinstance(Normalize, int):
        assert 0 <= Normalize <= 2
        NormalizeInt = Normalize
    else:
        raise TypeError("Normalize must be int or str")

    # Accept float for Radius
    if isinstance(Radius, float):
        assert 0 <= Radius
    else:
        raise TypeError("Radius must be float")

    return _direct_interpolator(
        x,
        y,
        xq,
        arg_num=1,
        Neighbours=0,
        Power=0.0,
        BasisFunction=0,
        StationarityFunction=0,
        NuggetEffect=0,
        Pulsation=0,
        InterpMethod=InterpMethodInt,
        Radius=Radius,
        Normalize=NormalizeInt,
    )


def ARBF(x, y, xq, InterpMethod="linear", Radius=0.0, Normalize=0):
    """
    ARBF interpolator function.

    Called by the user providing necessary input data. This function will check the format of the input arguments and
    call ``QuasarPy._direct_interpolator``.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Query data input, from which to make predictions.
    InterpMethod: int or str, default 'linear'
        Define a radial basis function (type of interpolation)

        0: 'multiquadratic',
        1: 'invMultiquadratic',
        2: 'thinPlateSpline',
        3: 'gaussian',
        4. 'linear'.

    Radius: float, default 0.0

        0.0: to be automatically calculated from data.

    Normalize: int, default 0
        Type of normalization to be performed

        0: No normalization,
        1: Min/Max normalization,
        2: Variance normalization.

    Returns
    -------
    numpy.ndarray
        Output predicitons corresponding to ``xq``, extracted from ``QuasarPy._direct_interpolator``.
    """
    # Accept integer or string for BasisFunction
    if isinstance(InterpMethod, str):
        InterpMethodMap = {
            "multiquadric": 0,
            "invMultiquadric": 1,
            "thinPlateSpline": 2,
            "gaussian": 3,
            "linear": 4,
        }
        InterpMethodInt = InterpMethodMap[InterpMethod]
    elif isinstance(InterpMethod, int):
        assert 0 <= InterpMethod <= 4
        InterpMethodInt = InterpMethod
    else:
        raise TypeError("InterpMethod must be int or str")

    # Accept integer or string for Normalize
    if isinstance(Normalize, str):
        NormalizeMap = {"No": 0, "Min/Max": 1, "Variance": 2}
        NormalizeInt = NormalizeMap[Normalize]
    elif isinstance(Normalize, int):
        assert 0 <= Normalize <= 2
        NormalizeInt = Normalize
    else:
        raise TypeError("Normalize must be int or str")

    # Accept float for Radius
    if isinstance(Radius, float):
        assert 0 <= Radius
    else:
        raise TypeError("Radius must be float")

    return _direct_interpolator(
        x,
        y,
        xq,
        arg_num=1,
        Neighbours=0,
        Power=0.0,
        BasisFunction=0,
        StationarityFunction=0,
        NuggetEffect=0,
        Pulsation=0,
        InterpMethod=InterpMethodInt,
        Radius=Radius,
        Normalize=NormalizeInt,
    )


def _POD_interpolator(
    x, y, xq, arg_num, Param1, Param2, NuggetEffect, Pulsation, Normalize, Save_Files, Modes, Tolerance, Rda_Flag
):
    """
    Function to be called by any POD interpolator method.

    Each of the POD interpolators calling this function will provide the value of the arguments to be used and set the
    rest to 0.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    arg_num: int
        Specify kind of direct interpolation to be performed.
        1: RBF.
        2: kriging.
        4: ARBF.
        5: INVD.
    Param1: int.
        Refers to different parameters depending on whihc interpolation method is used/
        RBF / ARBF: InterpMethod.
        Kriging: BasisFunction.
        INVD: Neighbours.
    Param2: int.
        Refers to different parameters depending on whihc interpolation method is used/
        RBF / ARBF: Radius.
        Kriging: Stationarityfucntion.
        INVD: Power.
    NuggetEffect: float
        Real values used for trigonometic radial basis function.
        0.0: not used.
    Pulsation: float
        Real values used for trigonometic radial basis function.
        0.0: not used.
    Normalize: int, {0, 1, 2}
        Type of normalization to be performed
        0: No normalization
        1: Min/Max normalization.
        2: Variance normalization.
    Save_Files: int.
        Flag to generate decomposition files.
        0: no storage.
        1: write as ASCII.
        -1: write as binary.
        2: read ASCII files.
        -2: read binary file.
    Modes: int.
        Number of active modes.
    Tolerance: float.
        Real parameter to calculate automatically the mode number to be kept.
    Rda_flag: int.
        Option to generate many files about decomposition.
        0: no files generated.
        1: files generated.

    Returns
    -------
    yq: numpy.ndarray
        Output predicitons corresponding to ``xq``.
    """
    input_data = {"X": x, "Y": y, "XQ": xq}

    output_keys = ["YQ"]

    if arg_num == 2:  # Kriging
        Param2Type = "int"
    else:
        Param2Type = "double"

    # fmt: off
    quasar_script = f"""
    Matrix train_X=loadCsv(cwd+"\X.csv")
    Matrix train_Y=loadCsv(cwd+"\Y.csv")
    Matrix query_X=loadCsv(cwd+"\XQ.csv")
    int arg_num = {arg_num}
    int Param1={Param1} #InterpMethod / BasisFunction / 0 / Neighbours
    {Param2Type} Param2={Param2} #Radius / StationarityFunction / 0 / Power
    double NuggetEffect={NuggetEffect}
    double Pulsation={Pulsation}
    int Normalize={Normalize}
    int Save_Files = {Save_Files}
    int Modes = {Modes}
    double Tolerance = {Tolerance}
    int Rda_Flag = {Rda_Flag}
    String Prefix
    Prefix = "YQ"
    Matrix YN
    YN=ext("interpolator", "matrix", "pod_all", train_X, train_Y, query_X,  arg_num, Param1, Param2, Normalize, NuggetEffect, Pulsation, Save_Files, Modes, Tolerance, Rda_Flag, Prefix)
    YN.saveCsv(cwd+"\YQ.csv","%16.8E") #output csv file
    """  # noqa

    # fmt: on

    return _run_quasar(input_data, output_keys, quasar_script)


def POD_kriging(
    x,
    y,
    xq,
    Normalize=0,
    Save_Files="No",
    Modes=0,
    Tolerance=0.0,
    Rda_Flag="No",
    BasisFunction="linear",
    StationarityFunction="h1",
    NuggetEffect=None,
    Pulsation=None,
):
    """
    POD Kriging interpolator function.

    Called by the user providing necessary input data. This function will check the format of the input arguments and
    call ``QuasarPy._POD_interpolator``.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    Normalize: int, default 0
        Type of normalization to be performed

        0: No normalization,
        1: Min/Max normalization,
        2: Variance normalization.

    Save_Files: int or str, default 'No'
        Flag to generate decomposition files.

        0: no storage,
        1: write as ASCII,
        -1: write as binary,
        2: read ASCII files,
        -2: read binary file.

    Modes: int, default 0
        Number of active modes.
    Tolerance: float, default 0.0
        Real parameter to calculate automatically the mode number to be kept.
    Rda_flag: int or str, default 'No'
        Option to generate many files about decomposition.

        0: no files generated,
        1: files generated.

    BasisFunction: int or str, default 'linear'
        Define basis function (type of interpolation).

        0: 'none',
        1: 'constant',
        2: 'linear',
        3: 'quadratic',
        4: 'cubic',
        5: 'trigonometric'.

    StationarityFunction: int or str, default 'h1'
        Define stationarity function (capacity of a response function to change its behavior).

        0: 'h1',
        1: 'h2',
        2: 'h3',
        3: 'exp',
        4: 'matern32'.

    NuggetEffect: float, default 0.0
        Real values used for trigonometic radial basis function.

        0.0: not used.

    Pulsation: float, default 0.0
        Real values used for trigonometic radial basis function.

        0.0: not used.

    Returns
    -------
    numpy.ndarray
        Output predicitons corresponding to ``xq``, extracted from ``QuasarPy._POD_interpolator``.
    """
    # Accept integer or string for BasisFunction
    if isinstance(BasisFunction, str):
        BasisFunctionMap = {"none": 0, "constant": 1, "linear": 2, "quadratic": 3, "cubic": 4, "trigonometric": 5}
        BasisFunctionInt = BasisFunctionMap[BasisFunction]
    elif isinstance(BasisFunction, int):
        assert 0 <= BasisFunction <= 5
        BasisFunctionInt = BasisFunction
    else:
        raise TypeError("BasisFunction must be int or str")

    # Accept integer or string for StationarityFunction
    if isinstance(StationarityFunction, str):
        StationarityFunctionMap = {"h1": 0, "h2": 1, "h3": 2, "exp": 3, "matern32": 4}
        StationarityFunctionInt = StationarityFunctionMap[StationarityFunction]
    elif isinstance(StationarityFunction, int):
        assert 0 <= StationarityFunction <= 4
        StationarityFunctionInt = StationarityFunction
    else:
        raise TypeError("StationarityFunction must be int or str")

    if NuggetEffect is None:
        NuggetEffect = 0.0

    if Pulsation is None:
        Pulsation = 0.0

    if BasisFunctionInt == 5:
        assert NuggetEffect > 0.0
        assert Pulsation > 0.0

    # Accept integer or string for Normalize
    if isinstance(Normalize, str):
        NormalizeMap = {"No": 0, "Min/Max": 1, "Variance": 2}
        NormalizeInt = NormalizeMap[Normalize]
    elif isinstance(Normalize, int):
        assert 0 <= Normalize <= 2
        NormalizeInt = Normalize
    else:
        raise TypeError("Normalize must be int or str")

    # Accept integer or string for Save_Files
    if isinstance(Save_Files, str):
        Save_FilesMap = {"No": 0, "Write as ASCII": 1, "Read ASCII": 2, "Write as Binary": -1, "Read Binary": -2}
        Save_FilesInt = Save_FilesMap[Save_Files]
    elif isinstance(Save_Files, int):
        assert -2 <= Save_Files <= 2
        Save_FilesInt = Save_Files
    else:
        raise TypeError("Save_Files must be int or str")

    # Accept integer for Modes
    if isinstance(Modes, int):
        assert 0 <= Modes <= len(y)
    else:
        raise TypeError("Modes must be int")

    # Accept float for Tolerance
    if isinstance(Tolerance, float):
        assert 0 <= Tolerance <= len(y)
    else:
        raise TypeError("Tolerance must be float")

    # Accept integer or string for Rda_Flag
    if isinstance(Rda_Flag, str):
        Rda_FlagMap = {"No": 0, "Yes": 1}
        Rda_FlagInt = Rda_FlagMap[Rda_Flag]
    elif isinstance(Rda_Flag, int):
        assert 0 <= Rda_Flag <= 1
        Rda_FlagInt = Rda_Flag
    else:
        raise TypeError("Rda_Flag must be int or str")

    return _POD_interpolator(
        x,
        y,
        xq,
        2,
        BasisFunctionInt,
        StationarityFunctionInt,
        NuggetEffect,
        Pulsation,
        NormalizeInt,
        Save_FilesInt,
        Modes,
        Tolerance,
        Rda_FlagInt,
    )


def POD_RBF(
    x,
    y,
    xq,
    Normalize=1,
    Save_Files="No",
    Modes=0,
    Tolerance=0.0,
    Rda_Flag="No",
    InterpMethod="multiquadric",
    Radius=0.0,
):
    """
    POD RBF interpolator function.

    Called by the user providing necessary input data. This function will check the format of the input arguments and
    call ``QuasarPy._POD_interpolator``.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    Normalize: int, default 1
        Type of normalization to be performed.

        0: No normalization,
        1: Min/Max normalization,
        2: Variance normalization.

    Save_Files: int or str, default 'No'
        Flag to generate decomposition files.

        0: no storage,
        1: write as ASCII,
        -1: write as binary,
        2: read ASCII files,
        -2: read binary file.

    Modes: int, default 0
        Number of active modes.
    Tolerance: float, default 0.0.
        Real parameter to calculate automatically the mode number to be kept.
    Rda_flag: int or str, default 'No'
        Option to generate many files about decomposition.

        0: no files generated,
        1: files generated.

    InterpMethod: int or str, default 'multiquadratic'
        Define a radial basis function (type of interpolation)

        0: 'multiquadratic',
        1: 'invMultiquadratic',
        2: 'thinPlateSpline',
        3: 'gaussian',
        4. 'linear'.

    Radius: float, default 0.0

        0.0: to be automatically calculated from data.

    Returns
    -------
    numpy.ndarray
        Output predicitons corresponding to ``xq``, extracted from ``QuasarPy._POD_interpolator``.
    """
    # Accept integer or string for BasisFunction
    if isinstance(InterpMethod, str):
        InterpMethodMap = {
            "multiquadric": 0,
            "invMultiquadric": 1,
            "thinPlateSpline": 2,
            "gaussian": 3,
            "linear": 4,
        }
        InterpMethodInt = InterpMethodMap[InterpMethod]
    elif isinstance(InterpMethod, int):
        assert 0 <= InterpMethod <= 4
        InterpMethodInt = InterpMethod
    else:
        raise TypeError("InterpMethod must be int or str")

    # Accept integer or string for Normalize
    if isinstance(Normalize, str):
        NormalizeMap = {"No": 0, "Min/Max": 1, "Variance": 2}
        NormalizeInt = NormalizeMap[Normalize]
    elif isinstance(Normalize, int):
        assert 0 <= Normalize <= 2
        NormalizeInt = Normalize
    else:
        raise TypeError("Normalize must be int or str")

    # Accept float for Radius
    if isinstance(Radius, float):
        assert 0 <= Radius
    else:
        raise TypeError("Radius must be float")

    # Accept integer or string for Save_Files
    if isinstance(Save_Files, str):
        Save_FilesMap = {"No": 0, "Write as ASCII": 1, "Read ASCII": 2, "Write as Binary": -1, "Read Binary": -2}
        Save_FilesInt = Save_FilesMap[Save_Files]
    elif isinstance(Save_Files, int):
        assert -2 <= Save_Files <= 2
        Save_FilesInt = Save_Files
    else:
        raise TypeError("Save_Files must be int or str")

    # Accept integer for Modes
    if isinstance(Modes, int):
        assert 0 <= Modes <= len(y)
    else:
        raise TypeError("Modes must be int")

    # Accept float for Tolerance
    if isinstance(Tolerance, float):
        assert 0 <= Tolerance <= len(y)
    else:
        raise TypeError("Tolerance must be float")

    # Accept integer or string for Rda_Flag
    if isinstance(Rda_Flag, str):
        Rda_FlagMap = {"No": 0, "Yes": 1}
        Rda_FlagInt = Rda_FlagMap[Rda_Flag]
    elif isinstance(Rda_Flag, int):
        assert 0 <= Rda_Flag <= 1
        Rda_FlagInt = Rda_Flag
    else:
        raise TypeError("Rda_Flag must be int or str")

    return _POD_interpolator(
        x,
        y,
        xq,
        1,
        Param1=InterpMethodInt,
        Param2=Radius,
        Normalize=NormalizeInt,
        Save_Files=Save_FilesInt,
        Modes=Modes,
        Tolerance=Tolerance,
        Rda_Flag=Rda_FlagInt,
        NuggetEffect=0,
        Pulsation=0,
    )


def POD_INVD(x, y, xq, Normalize=1, Save_Files="No", Modes=0, Tolerance=0.0, Rda_Flag="No", Neighbours=3, Power=2.0):
    """
    POD INVD interpolator function.

    Called by the user providing necessary input data. This function will check the format of the input arguments and
    call ``QuasarPy._POD_interpolator``.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    Normalize: int, default 0
        Type of normalization to be performed

        0: No normalization,
        1: Min/Max normalization,
        2: Variance normalization.

    Save_Files: int or str, default 'No'
        Flag to generate decomposition files.

        0: no storage,
        1: write as ASCII,
        -1: write as binary,
        2: read ASCII files,
        -2: read binary file.

    Modes: int, default 0
        Number of active modes.
    Tolerance: float, default 0.0
        Real parameter to calculate automatically the mode number to be kept.
    Rda_flag: int or str, default 'No'
        Option to generate many files about decomposition.

        0: no files generated,
        1: files generated.

    Neighbours: int, default 3
        Number of neighbours to reference.
    Power: float, {0, 0.25, 0.5, 2.0, 4.0}, default 2.0
        Order of interpoaltor function.

    Returns
    -------
    numpy.ndarray
        Output predicitons corresponding to ``xq``, extracted from ``QuasarPy._POD_interpolator``.
    """
    # Accept integer for Modes
    if isinstance(Neighbours, int):
        assert 0 <= Neighbours <= len(x)
    else:
        raise TypeError("Neighbours must be int")

    # Accept float for Power
    if isinstance(Power, float):
        assert Power in [0.0, 0.5, 0.25, 2.0, 4.0]
    else:
        raise TypeError("Power must be float")

    # Accept integer or string for Normalize
    if isinstance(Normalize, str):
        NormalizeMap = {"No": 0, "Min/Max": 1, "Variance": 2}
        NormalizeInt = NormalizeMap[Normalize]
    elif isinstance(Normalize, int):
        assert 0 <= Normalize <= 2
        NormalizeInt = Normalize
    else:
        raise TypeError("Normalize must be int or str")

    # Accept integer or string for Save_Files
    if isinstance(Save_Files, str):
        Save_FilesMap = {"No": 0, "Write as ASCII": 1, "Read ASCII": 2, "Write as Binary": -1, "Read Binary": -2}
        Save_FilesInt = Save_FilesMap[Save_Files]
    elif isinstance(Save_Files, int):
        assert -2 <= Save_Files <= 2
        Save_FilesInt = Save_Files
    else:
        raise TypeError("Save_Files must be int or str")

    # Accept integer for Modes
    if isinstance(Modes, int):
        assert 0 <= Modes <= len(y)
    else:
        raise TypeError("Modes must be int")

    # Accept float for Tolerance
    if isinstance(Tolerance, float):
        assert 0 <= Tolerance <= len(y)
    else:
        raise TypeError("Tolerance must be float")

    # Accept integer or string for Rda_Flag
    if isinstance(Rda_Flag, str):
        Rda_FlagMap = {"No": 0, "Yes": 1}
        Rda_FlagInt = Rda_FlagMap[Rda_Flag]
    elif isinstance(Rda_Flag, int):
        assert 0 <= Rda_Flag <= 1
        Rda_FlagInt = Rda_Flag
    else:
        raise TypeError("Rda_Flag must be int or str")

    return _POD_interpolator(
        x,
        y,
        xq,
        5,
        Param1=Neighbours,
        Param2=Power,
        Normalize=NormalizeInt,
        Save_Files=Save_FilesInt,
        Modes=Modes,
        Tolerance=Tolerance,
        Rda_Flag=Rda_FlagInt,
        NuggetEffect=0,
        Pulsation=0,
    )


def POD_ARBF(x, y, xq, Normalize=1, Save_Files="No", Modes=0, Tolerance=0.0, Rda_Flag="No"):
    """
    POD ARBF interpolator function.

    Called by the user providing necessary input data. This function will check the format of the input arguments and
    call ``QuasarPy._POD_interpolator``.

    Parameters
    ----------
    x: numpy.ndarray
        Training data input.
    y: numpy.ndarray
        Training data output.
    xq: numpy.ndarray
        Validation data input, from which to make predictions.
    Normalize: int, default 1
        Type of normalization to be performed

        0: No normalization,
        1: Min/Max normalization,
        2: Variance normalization.

    Save_Files: int or str, default 'No'
        Flag to generate decomposition files.

        0: no storage,
        1: write as ASCII,
        -1: write as binary,
        2: read ASCII files,
        -2: read binary file.

    Modes: int, default 0
        Number of active modes.
    Tolerance: float, default 0.0
        Real parameter to calculate automatically the mode number to be kept.
    Rda_flag: int or str, default 'No'
        Option to generate many files about decomposition.

        0: no files generated.
        1: files generated.

    Returns
    -------
    numpy.ndarray
        Output predicitons corresponding to ``xq``, extracted from ``QuasarPy._POD_interpolator``.
    """
    # Accept integer or string for Normalize
    if isinstance(Normalize, str):
        NormalizeMap = {"No": 0, "Min/Max": 1, "Variance": 2}
        NormalizeInt = NormalizeMap[Normalize]
    elif isinstance(Normalize, int):
        assert 0 <= Normalize <= 2
        NormalizeInt = Normalize
    else:
        raise TypeError("Normalize must be int or str")

    # Accept integer or string for Save_Files
    if isinstance(Save_Files, str):
        Save_FilesMap = {"No": 0, "Write as ASCII": 1, "Read ASCII": 2, "Write as Binary": -1, "Read Binary": -2}
        Save_FilesInt = Save_FilesMap[Save_Files]
    elif isinstance(Save_Files, int):
        assert -2 <= Save_Files <= 2
        Save_FilesInt = Save_Files
    else:
        raise TypeError("Save_Files must be int or str")

    # Accept integer for Modes
    if isinstance(Modes, int):
        assert 0 <= Modes <= len(y)
    else:
        raise TypeError("Modes must be int")

    # Accept float for Tolerance
    if isinstance(Tolerance, float):
        assert 0 <= Tolerance <= len(y)
    else:
        raise TypeError("Tolerance must be float")

    # Accept integer or string for Rda_Flag
    if isinstance(Rda_Flag, str):
        Rda_FlagMap = {"No": 0, "Yes": 1}
        Rda_FlagInt = Rda_FlagMap[Rda_Flag]
    elif isinstance(Rda_Flag, int):
        assert 0 <= Rda_Flag <= 1
        Rda_FlagInt = Rda_Flag
    else:
        raise TypeError("Rda_Flag must be int or str")

    return _POD_interpolator(
        x,
        y,
        xq,
        4,
        0,
        0.0,
        NuggetEffect=0,
        Pulsation=0,
        Normalize=NormalizeInt,
        Save_Files=Save_FilesInt,
        Modes=Modes,
        Tolerance=Tolerance,
        Rda_Flag=Rda_FlagInt,
    )


def DOE_generation(keyword, Nsites, Nseed, mat_MINMAX):
    """
    Generate and return a DoE as a numpy array.

    Parameters
    ----------
    keyword: str, {'box_behnken', 'grid', 'halton', 'hammersley', 'improved_latin_hypercube', 'latin_hypercube', 'optimal_latin_hypercube', 'minmax', 'montecarlo', 'normal', 'a_optimal', 'full_factorial', 'voronoi'}
        Method/distribution used for points generation.
    Nsites: int
        Number of points to generate.
    Nseed: int
        Random seed.
    mat_MINMAX: pandas.DataFrame
        DataFrame containing the name and range of the parameters to be used to create the DoE file.

    Return
    ------
    yq: numpy.ndarray
        Set of DOE points.

    Notes
    -----
    The rest of the parameters DoE generator takes as input are set to default value:

    >>> if keyword == 'voronoi':
    >>>     NoM = 1
    >>> elif (keyword == 'a_optimal' or keyword == 'full_factorial'):
    >>>     NoM = 10
    >>> else
    >>>     NoM = 3
    >>> Kscale = 0
    >>> rmin = 100.0
    >>> rmax = 200.0

    """  # noqa
    # Assert the given keyword is one of the possible options
    methods = [
        "box_behnken",
        "grid",
        "halton",
        "hammersley",
        "improved_latin_hypercube",
        "latin_hypercube",
        "optimal_latin_hypercube",
        "minmax",
        "montecarlo",
        "normal",
        "a_optimal",
        "full_factorial",
        "voronoi",
    ]
    assert keyword in methods

    # Accept integer for Nsites
    if isinstance(Nsites, int):
        assert 0 <= Nsites
    else:
        raise TypeError("Nsites must be int")

    # Accept integer for Nseed
    if isinstance(Nseed, int):
        assert 0 <= Nseed
    else:
        raise TypeError("Nseed must be int")

    # Assert mat_MINMAX contains m parameters

    Nvariables = len(mat_MINMAX)

    if keyword == "voronoi":
        NoM = 1
    elif keyword == "a_optimal" or keyword == "full_factorial":
        NoM = 10
    else:
        NoM = 3
    Kscale = 0
    rmin = 100.0
    rmax = 200.0

    input_data = {"mat_MINMAX": mat_MINMAX}

    output_keys = ["YQ"]
    # fmt: off
    quasar_script = f"""
    Matrix mat_MINMAX=loadCsv(cwd+"\mat_MINMAX.csv")
    int Nvariables={Nvariables}
    int Nsites={Nsites}
    int NoM={NoM}
    int Nseed={Nseed}
    int Kscale={Kscale}
    double rmin={rmin}
    double rmax={rmax}
    String output = (cwd+"\YQ.csv")
    Matrix OP
    OP=ext("doe", "{keyword}", Nvariables, Nsites, NoM, Nseed, Kscale, output, mat_MINMAX, rmin, rmax)
    OP.saveCsv(cwd+"\YQ.csv","%16.8E") #output csv file
    """  # noqa

    # fmt: on
    return _run_quasar(input_data, output_keys, quasar_script)


def DOE_scaling(mat_A, mat_MINMAX, Kscale=3):
    """
    Scale an input DoE file between a min and max values given.

    Parameters
    ----------
    mat_A: numpy.ndarray
        Data to be scaled.
    mat_MINMAX: pandas.DataFrame
        DataFrame containing the bounds for projection of DOE (new range).
    Kscale: int, {0, 1, 2, 3}, default=3

    Return
    ------
    yq: numpy.ndarray
        Set of scaled points.

    Notes
    -----
    The rest of the parameters are set to default value:

    >>> if Kscale == 3:
    >>>     rmin = 100.0
    >>>     rmax = 200.0
    >>> else:
    >>>     rmin = 0.0
    >>>     rmax = 0.0

    """
    # Accept integer for Kscale
    if isinstance(Kscale, int):
        assert 0 <= Kscale <= 3
    else:
        raise TypeError("Kscale must be int")

    # Assert mat_MINMAX is 2 x m
    # assert len(mat_MINMAX) == 2
    # assert len(mat_MINMAX[0]) == len(mat_A)
    if Kscale == 3:
        rmin = 100.0
        rmax = 200.0
    else:
        rmin = 0.0
        rmax = 0.0
    # Accept float for rmin
    assert isinstance(rmin, float)

    # Accept float for rmax
    assert isinstance(rmax, float)

    input_data = {"mat_A": mat_A, "mat_MINMAX": mat_MINMAX}

    output_keys = ["YQ"]

    # fmt: off
    quasar_script = f"""
    Matrix mat_A=loadCsv(cwd+"\mat_A.csv")
    Matrix mat_MINMAX=loadCsv(cwd+"\mat_MINMAX.csv")
    int Kscale={Kscale}
    double rmin={rmin}
    double rmax={rmax}
    Matrix OP
    OP=ext("doe", "scale", mat_A, mat_MINMAX, Kscale, rmin, rmax)
    OP.saveCsv(cwd+"\YQ.csv","%16.8E") #output csv file
    """  # noqa

    # fmt: on
    yq = _run_quasar(input_data, output_keys, quasar_script)

    if yq is not None:
        return yq
    else:
        return "*message ERROR"


def DOE_improvement(mat_A, maxNewPoints, Nseed):
    """
    Add points to an input DoE file.

    Parameters
    ----------
    mat_A: numpy.ndarray
        Data to which add points.
    maxNewPoints: int
        Number of points to add to input file.
    Nseed: int
        Random seed.

    Return
    ------
    yq: numpy.ndarray
        Set of DOE points.
    """
    # Accept integer for Nseed
    if isinstance(maxNewPoints, int):
        assert 0 <= maxNewPoints
    else:
        raise TypeError("maxNewPoints must be int bigger than 0")

    # Accept integer for Nseed
    if isinstance(Nseed, int):
        assert 0 <= Nseed
    else:
        raise TypeError("Nseed must be int bigger than 0")

    input_data = {"mat_A": mat_A}

    output_keys = ["YQ"]
    # fmt: off
    quasar_script = f"""
    Matrix mat_A=loadCsv(cwd+"\mat_A.csv")
    int Nseed={Nseed}
    int maxNewPoints={maxNewPoints}
    String prefix = "improved_doe"
    Matrix OP
    OP=ext("doe", "improvev2", mat_A, maxNewPoints, Nseed, prefix)
    OP.saveCsv(cwd+"\YQ.csv","%16.8E") #output csv file
    """  # noqa

    # fmt: on
    yq = _run_quasar(input_data, output_keys, quasar_script)

    if yq is not None:
        if yq[0][0] == "       -NAN(IND)":
            new_instances = pd.DataFrame(yq[-maxNewPoints:])
            yq = pd.DataFrame(mat_A).append(new_instances)
        return yq
    else:
        return None


def DOE_scatterplot(mat_A):
    """
    Plot an input DoE file.

    Parameters
    ----------
    mat_A: numpy.ndarray
        Data to plot.

    Return
    ------
    ax_list: list of dictionaries
        List containing a dictionary per plot to create. The key of the first entry in the dictionary will be the x axis name and the value the content to plot in such axis, similarly for the second entry in the dictionary and the y axis.
    """  # noqa

    mat_A = pd.DataFrame(mat_A)

    ax_list = []
    for i in range(len(mat_A.columns)):
        for j in range(i + 1, len(mat_A.columns)):
            entry = {"parameter " + str(i + 1): mat_A[i], "parameter " + str(j + 1): mat_A[j]}
            ax_list.append(entry)
    return ax_list


def PCA(mat_A, kernel=0, biplot=0, IstandardizeY=1):
    """
    PCA transform of an input dataset.

    Parameters
    ----------
    mat_A: pandas.DataFrame
        Data to transform.
    kernel: int or str, default 0

        0: None,
        1: Polynomial,
        2: RBF,
        3: sigmoid.

    biplot: int, default 0
        Type of biplot to draw.

        0: distance biplot,
        1: correlation biplot.

    IstandardizeY: int, default 1

        0: no standardization (raw data).
        1: ((Y-Ym)/std)

    Return
    ------
    yq: numpy.ndarray
        Dataset containing the transformed data.
    """
    # Accept integer for kernel
    if isinstance(kernel, int):
        assert 0 <= kernel <= 3
    else:
        raise TypeError("kernel must be int")

    # Accept integer for biplot
    if isinstance(biplot, int):
        assert 0 <= biplot <= 3
    else:
        raise TypeError("biplot must be int")

    # Accept integer for IstandardizeY
    if isinstance(IstandardizeY, int):
        assert 0 <= IstandardizeY <= 1
    else:
        raise TypeError("IstandardizeY must be int")

    input_data = {"mat_A": mat_A}

    output_keys = ["YQ"]

    # fmt: off
    quasar_script = f"""
    Matrix mat_A=loadCsv(cwd+"\mat_A.csv")
    int kernel={kernel}
    int biplot={biplot}
    int IstandardizeY={IstandardizeY}
    String output = (cwd+"\YQ.csv")
    Matrix OP
    OP=ext("mining", "pca", mat_A, kernel, biplot, IstandardizeY, output)
    OP.saveCsv(cwd+"\YQ.csv","%16.8E") #output csv file
    """  # noqa

    # fmt: on
    return _run_quasar(input_data, output_keys, quasar_script)


def PCA_scatterplot(mat_A):
    """
    Plot a PCA representation of an input DoE file.

    Parameters
    ----------
    mat_A: numpy.ndarray
        Data to plot after PCA transformation on ``Quasar.PCA()``.

    Return
    ------
    ax_list: list of dictionaries
        List containing a dictionary per plot to create. The key of the first entry in the dictionary will be the x axis name and the value the content to plot in such axis, similarly for the second entry in the dictionary and the y axis.
    """  # noqa
    mat_A = pd.DataFrame(mat_A)

    mat_A = pd.DataFrame(PCA(mat_A))[: len(mat_A)]

    ax_list = []
    for i in range(len(mat_A.columns)):
        for j in range(i + 1, len(mat_A.columns)):
            entry = {"parameter " + str(i + 1): mat_A[i], "parameter " + str(j + 1): mat_A[j]}
            ax_list.append(entry)
    return ax_list


def GPNewPoints(X, Y, XG, newpoints=1, PCA=False, Y_column_idx=0):
    """
    Adaptive sampling approach to select and return a set of points with highest variance to be added to the original
    dataset.

    Parameters
    ----------
    X: numpy.ndarray
        Current input data.
    Y: numpy.ndarray
        Current output data.
    XG: numpy.ndarray
        Set of candidate points.
    newPoints: int, default 1
        Number of points to select from ``XG``.
    PCA: bool, default False
        To analyze points in PCA coordinates or original.
    Y_column_idx: int, default 0 (becomes 1 in Quasar)
        Which column of Y to use to analyze where to add new poitns.

    Return
    ------
    Xnew: numpy.ndarray
        Set of new optimal points selected from ``XG``.
    """
    PCAInt = {True: 1, False: 0}[PCA]

    input_data = {"X": X, "Y": Y, "XG": XG}
    output_keys = ["Xnew"]

    # fmt: off
    quasar_script = f"""
Matrix X = loadCsv(cwd+"\X.csv")
Matrix Y = loadCsv(cwd+"\Y.csv")
Matrix XC = X
Matrix YC = Y
Matrix XG = loadCsv(cwd+"\XG.csv")
XPOINT = ext("operator","matrix","gpnewpoints",X,Y,XC,YC,XG,0,{newpoints},{PCAInt},{Y_column_idx+1},"KGNEWPOINTS")
XPOINT.saveCsv(cwd+"\Xnew.csv", "%16.8E")
    """ # noqa
    # fmt: on

    return _run_quasar(input_data, output_keys, quasar_script)


def normalize(mat_A):
    """
    Function to rescale data values into a range of [0,1]

    Parameters
    ----------
    mat_A: numpy.ndarray
        Data to be scaled.

    Return
    ------
    yq: numpy.ndarray
        Scaled data.
    """

    input_data = {"mat_A": mat_A}

    output_keys = ["YQ"]

    # fmt: off
    quasar_script = f"""
    Matrix mat_A=loadCsv(cwd+"\mat_A.csv")
    Matrix OP
    OP=ext("operator", "matrix", "normalize", mat_A)
    OP.saveCsv(cwd+"\YQ.csv","%16.8E") #output csv file
    """  # noqa

    # fmt: on
    return _run_quasar(input_data, output_keys, quasar_script)


gp_new_points = GPNewPoints  # alias

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
