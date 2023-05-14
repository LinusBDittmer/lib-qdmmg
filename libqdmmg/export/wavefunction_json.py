'''

author : Linus Bjarne Dittmer

'''

import json
import numpy
import libqdmmg.general as gen

def export_to_json(wp, path):
    '''
    Exports a wavepacket to a JSON file. The following format is applied:

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

    Parameters
    ----------
    wp : libqdmmg.general.wavepacket.Wavepacket
        Wavepacket which is to be exported.
    path : str
        Filepath of the json file (with .json ending)
    '''
    json_dict = {}
    json_dict['type'] = 'Wavepacket'
    sim = wp.sim
    json_dict['timesteps'] = numpy.array((numpy.arange(sim.tsteps)*sim.tstep_val)).tolist()
    gaussians = wp.gaussians
    json_dict['number of gaussians'] = len(gaussians)
    for i, gaussian in enumerate(gaussians):
        coeffs = wp.gauss_coeff[i].tolist()
        gauss_dict = { 'coeffs' : coeffs }
        gauss_dict['centre'] = gaussian.centre.tolist()
        gauss_dict['momentum'] = gaussian.momentum.tolist()
        gauss_dict['phase'] = gaussian.phase.tolist()
        g_string = 'gaussian' + str(i)
        json_dict[g_string] = gauss_dict

    with open(path, 'w') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

def import_from_json(wp, path):
    json_dict = None
    with open(path, 'r') as f:
        json_dict = json.load(f)

    if json_dict['type'] != 'Wavepacket':
        raise gen.InvalidJSONFlagException(path)

    timesteps_wp = numpy.array(numpy.arange(wp.sim.tsteps)*wp.sim.tstep_val)
    timesteps_loaded = numpy.array(json_dict['timesteps'])
    tdiff = timesteps_wp - timesteps_loaded
    assert numpy.dot(tdiff, tdiff) < 10**-9, f"Loaded timesteps do not match provided timesteps."
    num_gaussian = json_dict['number of gaussians']
    coeffs = []
    for g in range(num_gaussian):
        g_dict = json_dict['gaussian' + str(g)]
        gaussian = gen.Gaussian(wp.sim)
        gaussian.centre = numpy.array(g_dict['centre'])
        gaussian.momentum = numpy.array(g_dict['momentum'])
        gaussian.phase = numpy.array(g_dict['phase'])

        ts = wp.sim.tsteps-1
        gaussian.d_centre[1:ts] = 0.5 * (gaussian.centre[2:] - gaussian.centre[:ts-1]) / wp.sim.tstep_val
        gaussian.d_momentum[1:ts] = 0.5 * (gaussian.momentum[2:] - gaussian.momentum[:ts-1]) / wp.sim.tstep_val
        gaussian.d_phase[1:ts] = 0.5 * (gaussian.phase[2:] - gaussian.phase[:ts-1]) / wp.sim.tstep_val
        gaussian.d_centre[0] = (gaussian.centre[1] - gaussian.centre[0]) / wp.sim.tstep_val
        gaussian.d_momentum[0] = (gaussian.momentum[1] - gaussian.momentum[0]) / wp.sim.tstep_val
        gaussian.d_phase[0] = (gaussian.phase[1] - gaussian.phase[0]) / wp.sim.tstep_val

        wp.bind_gaussian(gaussian, numpy.ones(wp.sim.tsteps))
        c = numpy.array(g_dict['coeffs'])
        coeffs.append(c)

    wp.gauss_coeff = numpy.array(coeffs)


if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.general as gen
    import libqdmmg.potential as pot

    s = sim.Simulation(20, 0.01, dim=1, generations=1, verbose=3)
    p = pot.HarmonicOscillator(s, numpy.ones(1))
    s.bind_potential(p)
    s.gen_wavefunction()

    export_to_json(s.previous_wavefunction, 'trial.json')
    wp = gen.Wavepacket(s)
    import_from_json(wp, 'trial.json')
    s.logger.info("Max Difference in Coefficients:")
    s.logger.info(numpy.max(wp.gauss_coeff - s.previous_wavefunction.gauss_coeff))
    s.logger.info("Max Difference in Centre of Gaussian 0:")
    s.logger.info(numpy.max(wp.gaussians[0].centre - s.previous_wavefunction.gaussians[0].centre))
    s.logger.info("Max Difference in Momentum of Gaussian 0:")
    s.logger.info(numpy.max(wp.gaussians[0].momentum - s.previous_wavefunction.gaussians[0].momentum))
    s.logger.info("Max Difference in Phase of Gaussian 0:")
    s.logger.info(numpy.max(wp.gaussians[0].phase - s.previous_wavefunction.gaussians[0].phase))
    s.logger.info("Max Difference in Centre of Gaussian 1:")
    s.logger.info(numpy.max(wp.gaussians[1].centre - s.previous_wavefunction.gaussians[1].centre))
    s.logger.info("Max Difference in Momentum of Gaussian 1:")
    s.logger.info(numpy.max(wp.gaussians[1].momentum - s.previous_wavefunction.gaussians[1].momentum))
    s.logger.info("Max Difference in Phase of Gaussian 1:")
    s.logger.info(numpy.max(wp.gaussians[1].phase - s.previous_wavefunction.gaussians[1].phase))


