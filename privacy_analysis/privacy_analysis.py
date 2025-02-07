from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition
from autodp import mechanism_zoo, transformer_zoo
from autodp import rdp_bank
from autodp import rdp_acct
from autodp import utils
import math
from scipy.optimize import minimize_scalar
import numpy as np

def rdp_to_approxdp(rdp, alpha_max=np.inf, BBGHS_conversion=True):
    # from RDP to approx DP
    # alpha_max is an optional input which sometimes helps avoid numerical issues
    # By default, we are using the RDP to approx-DP conversion due to BBGHS'19's Theorem 21
    # paper: https://arxiv.org/pdf/1905.09982.pdf
    # if you need to use the simpler RDP to approxDP conversion for some reason, turn the flag off

    def approxdp(delta):

        """
        approxdp outputs eps as a function of delta based on rdp calculations
        :param delta:
        :return: the \epsilon with a given delta
        """

        if delta < 0 or delta > 1:
            print("Error! delta is a probability and must be between 0 and 1")
        if delta == 0:
            return rdp(np.inf)
        else:
            def fun(x):  # the input the RDP's alpha
                if x <= 1:
                    return np.inf
                else:
                    if BBGHS_conversion:
                        return np.maximum(rdp(x) + np.log((x-1)/x) - (np.log(delta) + np.log(x))/(x-1), 0)
                    else:
                        return np.log(1 / delta) / (x - 1) + rdp(x)

            results = minimize_scalar(fun, method='Bounded', bracket=(1,2), bounds=(1, alpha_max) )
            if results.success:
                return results.fun
            else:
                # There are cases when certain \delta is not feasible.
                # For example, let p and q be uniform the privacy R.V. is either 0 or \infty and unless all \infty
                # events are taken cared of by \delta, \epsilon cannot be < \infty
                return np.inf
    return approxdp


class DPGoodRadius(Mechanism):
    def __init__(self, tolerance, sigma_radius, name='DPGoodRadius'):
        Mechanism.__init__(self)
        self.name = name
        self.params = {
            'tolerance': tolerance,
            'sigma_radius': sigma_radius
        }
        def RDP_GoodRadius(alpha):
            max_iterations = 2 * math.ceil(math.log2((np.sqrt(2)/2) / tolerance))
            binarysearch_per_iteration = rdp_bank.RDP_gaussian({'sigma': sigma_radius}, alpha)
            rdp_total = max_iterations * binarysearch_per_iteration
            return rdp_total
        self.propagate_updates(RDP_GoodRadius, type_of_update='RDP')

class DPTesting(Mechanism):
    def __init__(self, sigma_test, name='DPTesting'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'sigma_test':sigma_test}
        def RDP_Testing(alpha):
            return rdp_bank.RDP_gaussian({'sigma': sigma_test}, alpha)
        self.propagate_updates(RDP_Testing, 'RDP')

class DPAvg(Mechanism):
    def __init__(self, sigma_avg, name='DPAvg'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'sigma_avg':sigma_avg}
        def RDP_Avg(alpha):
            return rdp_bank.RDP_gaussian({'sigma': sigma_avg}, alpha)
        self.propagate_updates(RDP_Avg, 'RDP')

class compose_DPGoodRadius_DPTesting_DPAvg(Mechanism):
    def __init__(self, tolerance, sigma_radius, sigma_test, sigma_avg, hat_T, name='compose_DPGoodRadius_DPTesting_DPAvg'):

        Mechanism.__init__(self)
        self.name=name
        self.params = {
            'tolerance': tolerance,
            'sigma_radius': sigma_radius,
            'sigma_test': sigma_test,
            'sigma_avg': sigma_avg,
            'hat_T': hat_T
        }
        compose = transformer_zoo.Composition()
        dp_good_radius = DPGoodRadius(tolerance, sigma_radius)
        dp_testing = DPTesting(sigma_test)
        dp_avg = DPAvg(sigma_avg)
        mech = compose([dp_good_radius, dp_testing, dp_avg], [1, hat_T, hat_T+1])
        rdp_total = mech.RenyiDP
        self.propagate_updates(rdp_total, type_of_update='RDP')

class RDP_AdaDPSyn(Mechanism):
    def __init__(self, prob, T_max, tolerance, sigma_radius, sigma_test, sigma_avg, hat_T, name='RDP_AdaDPSyn'):
        Mechanism.__init__(self)
        self.name=name
        subsample = transformer_zoo.AmplificationBySampling(PoissonSampling=False)
        compose = transformer_zoo.Composition()
        mech = compose_DPGoodRadius_DPTesting_DPAvg(tolerance, sigma_radius, sigma_test, sigma_avg, hat_T)
        mech.neighboring = 'replace_one'
        composed_sampled_mech = compose([subsample(mech,prob)],[T_max])
        rdp_total = composed_sampled_mech.RenyiDP
        self.propagate_updates(rdp_total, type_of_update='RDP')


# params = {prob, T_max, tolerance, sigma_radius, sigma_test, sigma_avg, hat_T}
def DP_AdaDPSyn(params, delta):
    mech = RDP_AdaDPSyn(params['prob'], params['T_max'], params['tolerance'], params['sigma_radius'], params['sigma_test'], params['sigma_avg'], params['hat_T'])
    rdp_func = mech.RenyiDP
    approxdp = rdp_to_approxdp(rdp_func, alpha_max=500, BBGHS_conversion=True)
    return approxdp(delta)

if __name__ == "__main__":
    # Desired delta 1/|D|
    # MIT-D: 1/1561 MIT-G: 1/2953 AGNEWS: 1/120000 DBPEDIA: 1/49999 TREC: 1/5452
    delta = 1/1561
    # T_max values: MIT-D: 20*4 MIT-G: 20*4 AGNEWS: 100 DBPEDIA: 100 TREC: 15
    T_max = 80
    # subsample rate: MIT-D: 40/1561 MIT-G: 40/2953 AGNEWS: 20/30000 DBPEDIA: 20/3558 TREC: 40/835
    sample_rate = 40/1561
    tolerance = 0.1

    # Here you need to check and let the output of DP_AdaDPSyn to be less the your target epsilon
    privacy_params = {
        'prob': sample_rate,
        'T_max': T_max,
        'tolerance': tolerance,
        'sigma_radius': 15,
        'sigma_test': 5,
        'sigma_avg': 0.83,
        'hat_T': 1
    }
    print(DP_AdaDPSyn(privacy_params, delta=delta))
