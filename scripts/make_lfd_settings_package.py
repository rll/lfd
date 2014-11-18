import argparse
import os
import lfd

def make_settings_tree(src, dst):
    names = os.listdir(src)
    for name in names:
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        if os.path.isdir(srcname):
            make_settings_tree(srcname, dstname)
        elif name == 'settings.py':
            if not os.path.isdir(dst):
                os.makedirs(dst)
            open(dstname, 'a')
            open(os.path.join(dst, '__init__.py'), 'a')

def make_lfd_settings_package(lfd_settings_name):
    """ Makes the lfd_settings package.

    Makes the lfd_settings package with settings.py files with the same 
    subpackage and module structure as the lfd package. Only makes subpackages 
    and modules for which the corresponding one in the lfd package has a 
    settings.py file.
    """
    lfd_name = os.path.dirname(lfd.__file__)
    make_settings_tree(lfd_name, lfd_settings_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lfd_settings_name', type=str, help="Destination name for the lfd_settings package.")
    args = parser.parse_args()
    
    make_lfd_settings_package(args.lfd_settings_name)

if __name__ == '__main__':
    main()
