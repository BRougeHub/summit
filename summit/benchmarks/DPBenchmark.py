from summit.strategies.base import Transform
from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
import numpy as np
from scipy.integrate import solve_ivp


class DeprotectionBenchmark(Experiment):
    """Benchmark representing a deprotection reaction
    The reaction occurs in a plug flow reactor where residence time, concentration and temperature
    can be adjusted. Maximizing Space time yield (STY) and conversion are the objectives.
    Parameters
    ----------
    noise_level: float, optional
        The mean of the random noise added to the concentration measurements in terms of
        percent of the signal. Default is 0.
    Examples
    --------
    >>> b = DeprotectionBenchmark()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)
    Notes
    -----
    This benchmark relies on the kinetics observerd during internal testing. The mechanistic
    model is solved using algebraic equations obtained from solving the ODE for plug flow reactor. These
    concentrations are then used to calculate STY and Conversion.
    _
    """

    def __init__(self, noise_level=0, **kwargs):
        domain = self._setup_domain()
        super().__init__(domain)
        self.rng = np.random.default_rng()
        self.noise_level = noise_level

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = 'residence time in minutes'
        domain += ContinuousVariable(name='tau', 
                                     description=des_1, bounds = [5, 25])
        
        des_2 = 'Concentration of SM in M'
        domain += ContinuousVariable(name='C_SM', 
                                     description=des_2, bounds = [0.3, 1.2])

        des_3 = 'Reactor Temperature in C'
        domain += ContinuousVariable(name='T', 
                                     description=des_3, bounds = [150, 250])
        
        #Objectives

        des_4 = 'Space Time Yield in kg/L/hr'
        domain+= ContinuousVariable(name='STY', description=des_4, 
                                   bounds = [0, 100],
                                   is_objective=True, 
                                   maximize=True)

        des_5 = 'Conversion'
        domain += ContinuousVariable(name='Conv', description=des_5, 
                                     bounds=[0, 1], 
                                     is_objective=True, 
                                     maximize = True)


        return domain

    def _run(self, conditions, **kwargs):
        tau = float(conditions["tau"])
        C_SM = float(conditions["C_SM"])
        T = float(conditions["T"])
        conv, sty, res = self._integrate_equations(tau, C_SM, T)
        conditions[("Conv", "DATA")] = conv
        conditions[("STY", "DATA")] = sty
        return conditions, {}

    def _integrate_equations(self, tau, C_SM, T, **kwargs):
        
        c_out = C_SM*np.exp(-112.0225*np.exp(-1287.8/T)*tau)
        
        #Add measurement noise
        c_out += (c_out*self.rng.normal(scale = self.noise_level, size = 1))/100)
        
        #Make sure the concentration is not negative
        c_out = max(0, c_out)
        
        #Calculate objectives
        conv = float(1 - (c_out/C_SM))
        sty = float(C_SM*conv/tau/1000*60*144.172)
        
        
        return conv, sty, {}

    def _integrand(self, t, C, T):
        pass

    def to_dict(self, **kwargs):
        experiment_params = dict(noise_level=self.noise_level)
        return super().to_dict(**experiment_params)
