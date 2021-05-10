import subprocess, os, platform

filepath = os.path.abspath('/Users/milan/Documents/1-Milan/Avis_d_impot_2020_sur_les_revenus_2019.pdf')


def open_pdf(filepath):
    if platform.system() == 'Darwin':       # macOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':    # Windows
        os.startfile(filepath)
    else:                                   # linux variants
        subprocess.call(('xdg-open', filepath))


open_pdf(filepath)