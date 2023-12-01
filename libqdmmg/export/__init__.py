'''

author : Linus Bjarne Dittmer

This package regulates export and import of generated wavepackets to a custom .json format, which is built according to the following schematic:


{
    'type': 'Wavepacket',
    'timesteps': [0.0, ..., 1.0],                                   // In atomic seconds
    'number of gaussians': 2,                                       // Integer
    'gaussian0':                                                    // Name is gaussian[number], where [number] reaches from 0 to [number of gaussians], exclusive
    {
        'centre': [ [0.0, 0.0], [0.0, 0.1], ..., [1.0, 1.0] ],      // 2D array
        'momentum': [ [0.0, 0.0, [0.0, -0.1], ..., [0.0, -1.0] ],   // Analogous to centre
        'phase': [0.0, 0.1, ..., 10.0],                             // Array of floats
        'coeffs': [0.5, 0.5, ..., 0.5],                             // Analogous to momentum
    },
    'gaussian1':
    {
        ...
    }
}

This json can be exported using export_to_json and imported to a wavepacket using import_from_json.

'''

import libqdmmg.export.wavefunction_json as wj

def export_to_json(wp, path):
    '''
    This function exports a wavepacket wp to a json format which is located at (path). Note that path needs to contain the full file name, i. e. './example.json', not just 'example', however, both relative and absolute filepaths are allowed.

    Parameters
    ----------
    wp : libqdmmg.general.wavepacket.Wavepacket
        The wavepacket that is to be exported
    path : str
        The file to which the wavepacket is to be exported

    Raises
    ------
    OSError:
        If an error occurs in I/O.
    '''
    wj.export_to_json(wp, path)

def import_from_json(wp, path):
    '''
    This function imports a wavepacket from a json into the provided wavepacket instance.

    Parameters
    ----------
    wp : libqdmmg.general.wavepacket.Wavepacket
        Destination wavepacket into which the json file is loaded
    path : str
        Filepath from which the json file is read. Note that this needs to be the full file name (i. e. './example.json', not 'example'), but may be a relative filepath.

    Raises
    ------
    FileNotFoundError:
        If the file given in (path) does not exist.
    InvalidJSONFlagException:
        If the JSON file does not contain a Wavepacket
    AssertionError:
        If the provided JSON and Wavepacket's time signature do not match
    '''
    wj.import_from_json(wp, path)
