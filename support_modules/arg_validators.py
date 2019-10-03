import argparse
import datetime
from typing import List

from support_modules.misc import date_to_integer


def valid_intdate(d: str) -> int:
    """
    Validación del formato del argumento de entrada y conversión de su representación en string al integer deseado.
    :param d: Argumento de entrada.
    :return: Argumento transformado al tipo deseado.
    """
    try:
        assert (len(d) == 8), 'Wrong format'
        return int(d)
    except ValueError:
        msg = "intdate argument must be a formatted as YYYYMMDD but received: '{0}'.".format(d)
        raise argparse.ArgumentTypeError(msg)
    except AssertionError:
        msg = "intdate argument must be a formatted as YYYYMMDD but received: '{0}'.".format(d)
        raise argparse.ArgumentTypeError(msg)


def valid_predict_date(d: str) -> int:
    """
        Validación del formato del argumento de entrada y conversión de su representación en string al integer deseado.
        :param d: Argumento de entrada.
        :return: Argumento transformado al tipo deseado.
        """
    if d is None or d.lower() == 'none':
        today = datetime.date.today()
        predict_date = date_to_integer(today - datetime.timedelta(today.weekday() + 1))
        return predict_date
    else:
        return valid_intdate(d)


def valid_intdates(d: str) -> List[str]:
    """
    Validación del formato del argumento de entrada y conversión de su representación en string al integer deseado.
    :param d: Argumento de entrada.
    :return: Argumento transformado al tipo deseado.
    """
    err_msg = "intdates argument must be a formatted as [YYYYMMDD, YYYYMMDD, ...] where there can be as many YYYYMMDD " \
              "formatted dates as required. In case only one is inputted, the wraping [] can be ommited. " \
              "However, the argument received: '{0}'.".format(d)
    try:
        assert d[0] == '[' and d[-1] == ']', err_msg
        dates = d[1:-1].split(',')
        assert all([len(d_) == 8 for d_ in dates]), err_msg
        # noinspection PyStatementEffect
        [int(d_) == 8 for d_ in dates]
        return dates
    except ValueError:
        raise argparse.ArgumentTypeError(err_msg)


def valid_env(e):
    """
    Validación del entorno introducido de acuerdo a las opciones contempladas.
    :param e: Argumento de entrada.
    :return: Argumento validado.
    """

    try:
        e = e.lower()
        assert e in ('stg', 'rc', 'pro')
        return e
    except AssertionError:
        msg = "env must be one of [stg, rc, pro] and received: '{0}'.".format(e)
        raise argparse.ArgumentTypeError(msg)


def valid_mode(m):
    """
    Validación del modo introducido de acuerdo a las opciones contempladas.
    :param m: Argumento de entrada.
    :return: Argumento validado.
    """

    try:
        m = m.lower()
        assert m in ('train', 'predict')
        return m
    except AssertionError:
        msg = "mode must be one of [train, predict] and received: '{0}'.".format(m)
        raise argparse.ArgumentTypeError(msg)


def valid_validation_horizons(v):
    """
    Validación de los horizontes de validación.
    :param v: Argumento de entrada.
    :return: Argumento validado.
    """
    err_msg = 'months_ahead must be eiter a integer or a string formatted as: "[1,2,3]" to implement ' \
              'multiple validation horizons and received: "{0}".'.format(v)
    try:
        return [int(v)]
    except ValueError:
        assert v[0] == '[' and v[-1] == ']', err_msg
        v = v[1:-1].split(',')
        return [int(v_) for v_ in v]
    except:
        raise argparse.ArgumentTypeError(err_msg)


def valid_brand(b):
    """
    Validación del brand introducido de acuerdo a las opciones contempladas.
    :param b: Argumento de entrada.
    :return: Argumento validado.
    """
    valid_brands = ('claro', 'globo')

    try:
        b = b.lower()
        assert b in valid_brands
        return b
    except AssertionError:
        msg = "brand must be one of {0} and received: '{1}'.".format(valid_brands, b)
        raise argparse.ArgumentTypeError(msg)


def valid_int(i):
    """
    Validación del formato del argumento de entrada y conversión de su representación en string al entero deseado.
    :param i: Argumento de entrada.
    :return: Argumento transformado al tipo deseado.
    """
    try:
        return int(i)
    except ValueError:
        msg = "Argument must be integer and received: '{0}'.".format(i)
        raise argparse.ArgumentTypeError(msg)


def valid_bool(b):
    """
    Validación del formato y valor del argumento de entrada.
    :param b: Argumento de entrada.
    :return: El argumento covnertido en booleano.
    """
    b_low = b.lower()
    assert b_low in ('true', 'false'), \
        "Wrong format on boolean argument: accepted values are ['True', 'true', 'False', 'false'] but received: {}".format(
            b)
    return b_low == 'true'
