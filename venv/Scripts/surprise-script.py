#!C:\Users\IsabelleBernard\PycharmProjects\recommending_system\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'scikit-surprise==1.0.6','console_scripts','surprise'
__requires__ = 'scikit-surprise==1.0.6'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('scikit-surprise==1.0.6', 'console_scripts', 'surprise')()
    )
