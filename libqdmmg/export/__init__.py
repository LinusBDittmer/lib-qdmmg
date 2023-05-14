'''

author : Linus Bjarne Dittmer

'''

import libqdmmg.export.wavefunction_json as wj

def export_to_json(wp, path):
    wj.export_to_json(wp, path)

def import_from_json(wp, path):
    wj.import_from_json(wp, path)
