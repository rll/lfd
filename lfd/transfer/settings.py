import os

JOINT_LENGTH_PER_STEP = .1
FINGER_CLOSE_RATE     = .1

lfd_settings_name = os.environ.get('LFD_SETTINGS_PACKAGE')
if lfd_settings_name:
    import importlib
    lfd_settings = importlib.import_module(lfd_settings_name)
    try:
        from lfd_settings.transfer.settings import *
    except ImportError:
        pass
