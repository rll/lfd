from rapprentice.colorize import colorize
import subprocess

def call_and_print(cmd,color='green'):
    print colorize(cmd, color, bold=True)
    subprocess.check_call(cmd, shell=True)