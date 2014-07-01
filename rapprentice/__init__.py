import logging as _logging

LOG = _logging.getLogger("rapprentice")
LOG.setLevel(_logging.DEBUG)

_ch = _logging.StreamHandler()
_ch.setLevel(_logging.DEBUG)
_formatter = _logging.Formatter('%(levelname)s - %(message)s')
_ch.setFormatter(_formatter)
LOG.addHandler(_ch)

