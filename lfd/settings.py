import os

DEBUG = False

lfd_settings_name = os.environ.get('LFD_SETTINGS_PACKAGE')
if lfd_settings_name:
    import importlib
    lfd_settings = importlib.import_module(lfd_settings_name)
    try:
        from lfd_settings.settings import *
    except ImportError:
        pass
