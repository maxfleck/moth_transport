import numpy as np
np.testing.Tester = np.testing.TestCase
import pandas as pd
import json
import scipy
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from feos.si import * # SI numbers and constants
#from si_units import * # SI numbers and constants

from feos.pcsaft import *
from feos.eos import *


class MotH():
    """
    ...
    """
    def __init__(self, parameters, data=[], data_diffusion=[], 
                 data_tc=[], 
                 data_sos=[], 
                 p=[ 2,2, 1.4, .1, .1],
                 bounds=[[1,6],
                         [-1,1],[-.1,1],[-.1,1],
                         [-1,1],[-.1,1],[-.1,1]],
                 penalty_V_f=0.0,
                 penalty_E_act=1,
                 water_hbond=False,
                 ):
        self.p = p
        self.data = data
        self.parameters = parameters
        self.data_diffusion = data_diffusion
        self.data_tc = data_tc
        self.data_sos = data_sos
        self.dd = [1]
        self.p_tc = [1]
        self.p_sos = [1]
        self.water_hbond = water_hbond
        self.bounds = bounds
        self.penalty_V_f = penalty_V_f
        self.penalty_E_act = penalty_E_act
        if len(data_diffusion) > 0:
            self.diffusion_flag = True
        else:
            self.diffusion_flag = False
        if len(data_tc) > 0:
            self.tc_flag = True
        else:
            self.tc_flag = False
        if len(data_sos) > 0:
            self.sos_flag = True
        else:
            self.sos_flag = False                        
        return

    def predict_log_vf13(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        _,n0,n1,n2,_,_,_ = p
        n13 = 84446884.95844965
        vf13_0 =  data["V"]**(1/3) / n13
        x = -data["s_res"]/8.314
        x = n0*x + n1*x**2 + n2*x**3
        return np.log(vf13_0) +  x

    def predict_log_lambdas(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_,_,_,_ = p
        lambdas = 1/ (data["V"])
        return np.log(lambdas) - np.log(a**2)

    def predict_eyring_ref(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        a,_,_,_,_,_,_ = p

        const = np.log(7.227814978226432) # np.sqrt(2*np.pi*8.31446261815324)
        c0 = np.log( np.sqrt(data["M"]*data["temperature"]) )

        log_vf13 = self.predict_log_vf13(p=p,data=data)
        log_lambdas = self.predict_log_lambdas(p=p,data=data)

        return const+c0+log_vf13+log_lambdas

    def predict_energy_barrier(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        _,_,_,_,n0,n1,n2 = p
        x = -data["E_res"]/data["temperature"]/8.314
        #return n0*x + np.log( np.abs( n1*x**2 + n2*x**4) )
        ret =  n0*x + n1*x**2 + n2*x**3
        if self.water_hbond:
            ret = 6.3*1000/data["temperature"]/8.314
        return ret
    
    def predict(self,p=[],data=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        ey_ref = self.predict_eyring_ref(p=p,data=data)
        barrier = self.predict_energy_barrier(p=p,data=data)
        return ey_ref + barrier

    def predict_diffusion(self,p=[],data=[],dd=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        if len(dd)==0:
            dd=self.dd            
        log_dd = np.log(dd[0]/6)
        log_vis = self.predict(p=p,data=data)
        n13 = 84446884.95844965
        log_vf13 =  np.log(data["V"]**(1/3) / n13)
        log_kT = np.log( 1.380649e-23*data["temperature"] )
        return log_kT -log_vis -log_vf13 + log_dd
    
    def predict_thermal_conductivity(self,p=[],data=[],p_tc=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        if len(p_tc)==0:
            p_tc=self.p_tc            

        x = -data["s_res"]/8.314
        xx = p[1]*x + p[2]*x*x + p[3]*x*x*x

        R = 8.314
        kb = 1.380649e-23
        nav = 6.02214076e+23 
        #gamma = data["c_p"]/data["c_v"]
        #eucken = 3*p_tc[0]  / gamma
        #eucken = p_tc[0]*( 9*gamma - 5) / (  15*( gamma -1 ) )
        eucken = 4*p_tc[0]

        sq = np.sqrt( R*data["temperature"] / data["M"] )
        llog = np.log( 2* eucken * kb* ( nav / data["V"] )**(2/3) * sq ) 
        return llog + 1/4* xx

    def predict_speed_of_sound(self,p=[],data=[],p_sos=[]):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data
        if len(p_sos)==0:
            p_sos=self.p_sos         
        x = -data["s_res"]/8.314
        xx = p[1]*x + p[2]*x*x + p[3]*x*x*x

        R = 8.314
        gamma_tonks = ( data["c_v"] + 3*p_sos[0]*8.314  ) / data["c_v"]* p_sos[0]
        #gamma_tonks = 1* p_sos[0]
        c_gas =  np.sqrt( gamma_tonks * 3*R *data["temperature"] / data["M"] )
        xx = 3/8*xx 
        return np.log(c_gas*np.exp(xx)  )

    def error(self,p=[],data=[],y_key="viscosity"):
        if len(p)==0:
            p=self.p
        if len(data)==0:
            data=self.data     
        pred = self.predict(p=p,data=data)
        _,_,a,b,_,c,d = p
        p0 = self.penalty_V_f*( np.abs(a)+np.abs(b) )
        p1 = self.penalty_E_act*( np.abs(c)+np.abs(d) )
        error = np.mean( ( pred - np.log(data[y_key]) )**2 )
        return 1*(error + p0 + p1  ) *1e10

    def error_diffusion(self,dd=[],data_diffusion=[],y_key_diffsion="D",p=[]):
        if len(p)==0:
            p=self.p              
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion                 
        if len(dd)==0:
            dd=self.dd        
        pred = self.predict_diffusion(p=p,data=data_diffusion,dd=dd)
        error = np.sum( ( pred - np.log(data_diffusion[y_key_diffsion]) )**2 )
        return error*1e10
    
    def error_tc(self,p_tc=[],data_tc=[],y_key_tc="thermal_conductivity",p=[]):
        if len(p)==0:
            p=self.p              
        if len(data_tc)==0:
            data_tc=self.data_tc               
        if len(p_tc)==0:
            p_tc=self.p_tc        
        pred = self.predict_thermal_conductivity(p=p,data=data_tc,p_tc=p_tc)
        error = np.sum( ( pred - np.log(data_tc[y_key_tc]) )**2 )
        return error*1e10    
    
    def error_sos(self,p_sos=[],data_sos=[],y_key_sos="speed_of_sound_m_s",p=[]):
        if len(p)==0:
            p=self.p              
        if len(data_sos)==0:
            data_sos=self.data_sos               
        if len(p_sos)==0:
            p_sos=self.p_sos        
        pred = self.predict_speed_of_sound(p=p,data=data_sos,p_sos=p_sos)
        error = np.sum( ( pred - np.log(data_sos[y_key_sos]) )**2 )
        return error*1e10        
    
    def error_both(self,pdd=[], data=[],y_key="viscosity",
                        data_diffusion=[],y_key_diffsion="D"):
        if len(pdd)==0:
            p=self.p 
            dd=self.dd
        else:
            p=pdd[:-1]
            dd=[pdd[-1]]
        if len(data)==0:
            data=self.data                 
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion                 
    
        pred = self.predict(p=p,data=data)
        error = np.mean( ( pred - np.log(data[y_key]) )**2 )

        pred = self.predict_diffusion(p=p,data=data_diffusion,dd=dd)
        d_error = np.mean( ( pred - np.log(data_diffusion[y_key_diffsion]) )**2 )
        error += 1.5*d_error
        return error*1e10
    
    def error_full(self,pdd=[], data=[],y_key="viscosity",
                        data_diffusion=[],y_key_diffsion="D",
                        data_tc=[],y_key_tc="thermal_conductivity",
                        data_sos=[],y_key_sos="speed_of_sound_m_s",
                        ):
        if len(pdd)==0:
            p=self.p 
            dd=self.dd
            p_tc=self.p_tc
            p_sos=self.p_sos
        else:
            p=pdd[:-3]
            dd=[pdd[-3]]
            p_tc=[pdd[-2]]
            p_sos=[pdd[-1]]
        if len(data)==0:
            data=self.data                 
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion       
        if len(data_tc)==0:
            data_tc=self.data_tc                 
        if len(data_sos)==0:
            data_sos=self.data_sos                             
    
        error = 0

        pred = self.predict(p=p,data=data)
        v_error = np.mean( ( pred - np.log(data[y_key]) )**2 )
        error += 1.5*v_error

        pred = self.predict_diffusion(p=p,data=data_diffusion,dd=dd)
        d_error = np.mean( ( pred - np.log(data_diffusion[y_key_diffsion]) )**2 )
        error += 1.5*d_error

        pred = self.predict_thermal_conductivity(p=p,data=data_tc,p_tc=p_tc)
        tc_error = np.mean( ( pred - np.log(data_tc[y_key_tc]) )**2 )
        error += 1.5*tc_error

        pred = self.predict_speed_of_sound(p=p,data=data_tc,p_sos=p_sos)
        sos_error = np.mean( ( pred - np.log(data_sos[y_key_sos]) )**2 )
        error += 1.5*sos_error        

        return error*1e10

    def error_vtc(self,pdd=[], data=[],y_key="viscosity",
                        data_tc=[],y_key_tc="thermal_conductivity",
                        ):
        if len(pdd)==0:
            p=self.p 
            dd=self.dd
            p_tc=self.p_tc
            p_sos=self.p_sos
        else:
            p=pdd[:-1]
            p_tc=[pdd[-1]]
        if len(data)==0:
            data=self.data                    
        if len(data_tc)==0:
            data_tc=self.data_tc                 
                    
        error = 0

        pred = self.predict(p=p,data=data)
        v_error = np.mean( ( pred - np.log(data[y_key]) )**2 )
        error += 1.5*v_error

        pred = self.predict_thermal_conductivity(p=p,data=data_tc,p_tc=p_tc)
        tc_error = np.mean( ( pred - np.log(data_tc[y_key_tc]) )**2 )
        error += 1.5*tc_error
    
        return error*1e10



    def train(self,p=[],data=[],y_key="viscosity"):
        if len(p)==0:
            p=self.p       
        if len(data)==0:
            data=self.data            
        #bounds = [[0.1,40],[-10,15],[-10,10],[-1000,800],[-1000,800]]
        ferr = lambda x: self.error(x,data,y_key)
        res = minimize(ferr, p, bounds=self.bounds )
        self.p = res.x
        return res
    
    def train_diffusion(self,data_diffusion=[],y_key_diffsion="D"):
        dd=self.dd       
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion            
        bounds = [[0.01,3]]
        ferr = lambda x: self.error_diffusion(x,data_diffusion,y_key_diffsion)
        res = minimize(ferr, dd, bounds=bounds )
        self.dd = res.x
        return res
    
    def train_thermal_conductivity(self,data_tc=[],y_key_tc="thermal_conductivity"):
        dd=self.dd       
        if len(data_tc)==0:
            data_tc=self.data_tc            
        bounds = [[0.01,3]]
        ferr = lambda x: self.error_tc(x,data_tc,y_key_tc)
        res = minimize(ferr, dd, bounds=bounds )
        self.dd = res.x
        return res    

    def train_both(self,data=[],y_key="viscosity",
              data_diffusion=[],y_key_diffsion="D"):
        pdd=np.concat([self.p ,self.dd])
        if len(data)==0:
            data=self.data  
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion                         
        bounds = self.bounds.append([0.01,3])
        ferr = lambda x: self.error_both(x,data,y_key)
        res = minimize(ferr, pdd, bounds=bounds )

        self.p = res.x[:-1]
        self.dd = [res.x[-1]]
        return res

    def train_vtc(self, data=[], y_key="viscosity",
                        data_tc=[], y_key_tc="thermal_conductivity",
                        n_iter=5):
        if len(data) == 0:
            data = self.data
        if len(data_tc) == 0:
            data_tc = self.data_tc

        # L-BFGS-B options for tighter convergence
        opts = {"ftol": 1e-15, "gtol": 1e-8, "maxiter": 5000, "maxfun": 20000}

        for _ in range(n_iter):
            # Block A: optimize all 7 viscosity params (p_tc fixed)
            ferr_v = lambda x: self.error(x, data=data, y_key=y_key)
            res_v = minimize(ferr_v, list(self.p), bounds=self.bounds,
                            method="L-BFGS-B", options=opts)
            self.p = list(res_v.x)

            # Block B: optimize p_tc alone (1D, viscosity params fixed)
            ferr_tc = lambda pt: self.error_tc(
                [pt[0]], data_tc=data_tc, y_key_tc=y_key_tc, p=list(self.p)
            )
            res_tc = minimize(ferr_tc, list(self.p_tc),
                            bounds=[[0.00001, 300]],
                            method="L-BFGS-B", options=opts)
            self.p_tc = list(res_tc.x)

        # Final joint refinement (captures shared p[1:4] coupling)
        bounds = self.bounds + [[0.00001, 300]]
        pdd = list(self.p) + list(self.p_tc)
        ferr = lambda x: self.error_vtc(x, data=data, y_key=y_key,
                                        data_tc=data_tc, y_key_tc=y_key_tc)
        res = minimize(ferr, pdd, bounds=bounds, method="L-BFGS-B", options=opts)
        self.p = list(res.x[:-1])
        self.p_tc = [res.x[-1]]
        print(res)
        return res

    def train_full(self, data=[], y_key="viscosity",
                        data_diffusion=[], y_key_diffsion="D",
                        data_tc=[], y_key_tc="thermal_conductivity",
                        data_sos=[], y_key_sos="speed_of_sound_m_s",
                        n_iter=5):
        if len(data) == 0:          data = self.data
        if len(data_diffusion) == 0: data_diffusion = self.data_diffusion
        if len(data_tc) == 0:       data_tc = self.data_tc
        if len(data_sos) == 0:      data_sos = self.data_sos

        opts = {"ftol": 1e-15, "gtol": 1e-8, "maxiter": 5000, "maxfun": 20000}

        for _ in range(n_iter):
            # Block A: core params, viscosity only, all extra params fixed
            ferr_v = lambda x: self.error(x, data=data, y_key=y_key)
            res = minimize(ferr_v, list(self.p), bounds=self.bounds,
                        method="L-BFGS-B", options=opts)
            self.p = list(res.x)

            # Block B1: dd alone (1D, diffusion only)
            ferr_dd = lambda x: self.error_diffusion(
                [x[0]], data_diffusion=data_diffusion,
                y_key_diffsion=y_key_diffsion, p=list(self.p))
            res = minimize(ferr_dd, list(self.dd),
                        bounds=[[0.01, 3]], method="L-BFGS-B", options=opts)
            self.dd = list(res.x)

            # Block B2: p_tc alone (1D, TC only)
            ferr_tc = lambda x: self.error_tc(
                [x[0]], data_tc=data_tc, y_key_tc=y_key_tc, p=list(self.p))
            res = minimize(ferr_tc, list(self.p_tc),
                        bounds=[[0.00001, 300]], method="L-BFGS-B", options=opts)
            self.p_tc = list(res.x)

            # Block B3: p_sos alone (1D, SOS only)
            ferr_sos = lambda x: self.error_sos(
                [x[0]], data_sos=data_sos, y_key_sos=y_key_sos, p=list(self.p))
            res = minimize(ferr_sos, list(self.p_sos),
                        bounds=[[0.00001, 300]], method="L-BFGS-B", options=opts)
            self.p_sos = list(res.x)

        # Final joint refinement over all 10 params
        bounds = self.bounds + [[0.01, 3]] + [[0.00001, 300]] + [[0.00001, 300]]
        pdd = list(self.p) + list(self.dd) + list(self.p_tc) + list(self.p_sos)
        ferr = lambda x: self.error_full(x, data=data, y_key=y_key,
                        data_diffusion=data_diffusion, y_key_diffsion=y_key_diffsion,
                        data_tc=data_tc, y_key_tc=y_key_tc,
                        data_sos=data_sos, y_key_sos=y_key_sos)
        res = minimize(ferr, pdd, bounds=bounds, method="L-BFGS-B", options=opts)

        self.p    = list(res.x[:7])
        self.dd   = [res.x[7]]
        self.p_tc = [res.x[8]]
        self.p_sos= [res.x[9]]
        print(res)
        return res


    def train_fullx(self, data=[],y_key="viscosity",
                        data_diffusion=[],y_key_diffsion="D",
                        data_tc=[],y_key_tc="thermal_conductivity",
                        data_sos=[],y_key_sos="speed_of_sound_m_s",
                        ):
        if len(data)==0:
            data=self.data                 
        if len(data_diffusion)==0:
            data_diffusion=self.data_diffusion       
        if len(data_tc)==0:
            data_tc=self.data_tc                 
        if len(data_sos)==0:
            data_sos=self.data_sos   

        bounds = self.bounds + 3*[[0.00001,300]]

        pdd= self.p + self.dd + self.p_tc + self.p_sos
        ferr = lambda x: self.error_full( x, data=data,y_key=y_key,
                        data_diffusion=data_diffusion,y_key_diffsion=y_key_diffsion,
                        data_tc=data_tc,y_key_tc=y_key_tc,
                        data_sos=data_sos,y_key_sos=y_key_sos,)
        res = minimize(ferr, pdd, bounds=bounds )

        print(res.x)
        self.pp=res.x[:-3]
        self.dd=[res.x[-3]]
        self.p_tc=[res.x[-2]]
        self.p_sos=[res.x[-1]]        
        return res
    
         


def collision_integral( T, p):
    """
    computes analytical solution of the collision integral

    T: reduced temperature
    p: parameters

    returns analytical solution of the collision integral
    """
    A,B,C,D,E,F,G,H,R,S,W,P = p
    return A/T**B + C/np.exp(D*T) + E/np.exp(F*T) + G/np.exp(H*T) + R*T**B*np.sin(S*T**W - P)

def get_omega11(red_temperature):
    """
    computes analytical solution of the omega11 collision integral
    
    red_temperature: reduced temperature
    
    returns omega11
    """
    p11 = [ 
        1.06036,0.15610,0.19300,
        0.47635,1.03587,1.52996,
        1.76474,3.89411,0.0,
        0.0,0.0,0.0
    ]
    return collision_integral(red_temperature,p11)

def get_omega22(red_temperature):
    """
    computes analytical solution of the omega22 collision integral

    red_temperature: reduced temperature

    returns omega22
    """
    p22 = [ 
         1.16145,0.14874,0.52487,
         0.77320,2.16178,2.43787,
         0.0,0.0,-6.435/10**4,
         18.0323,-0.76830,7.27371
        ]
    return collision_integral(red_temperature,p22)

def get_viscosity_CE(temperature, saft_parameters):
    """
    computes viscosity reference for an array of temperatures
    uses pc-saft parameters

    temperature: array of temperatures
    saft_parameters: pc saft parameter object build with feos

    returns reference
    """
    epsilon = saft_parameters.pure_records[0].model_record.epsilon_k*KELVIN
    sigma   = saft_parameters.pure_records[0].model_record.sigma*ANGSTROM
    m       = saft_parameters.pure_records[0].model_record.m
    M       = saft_parameters.pure_records[0].molarweight*GRAM/MOL
    red_temperature = temperature/epsilon

    omega22 = get_omega22(red_temperature)

    sigma2 = sigma**2
    M_SI = M

    sq1  = np.sqrt( M_SI * KB * temperature / NAV /np.pi) # /METER**2 / KILOGRAM**2 *SECOND**2 ) *METER*KILOGRAM/SECOND
    div1 = omega22 * sigma2
    viscosity_reference = 5/16* sq1 / div1 #*PASCAL*SECOND
    viscosity_reference_m = 5/16* sq1 / div1/ m #*PASCAL*SECOND
    viscosity_reference_ig = 5/16* sq1 / sigma2
    return viscosity_reference, viscosity_reference_ig, viscosity_reference_m, omega22



def calc_stuff(sd, parameters):
    #print(sd)
    try:

        J_mol = JOULE/MOL
        J_molK = J_mol/KELVIN
        KG_m3 = KILO*GRAM/METER**3
        MOL_m3 = MOL/METER**3
        PS = PASCAL*SECOND
        
        eos = EquationOfState.pcsaft(parameters)
        M = parameters.pure_records[0].molarweight *(GRAM/MOL)
        m = parameters.pure_records[0].model_record.m
        epsilon = parameters.pure_records[0].model_record.epsilon_k*KELVIN
        sigma   = parameters.pure_records[0].model_record.sigma*ANGSTROM
        
        #sd = {"temperature":325*KELVIN, "pressure":2*BAR}
        if "pressure" in sd.keys():
            if sd["state"] == "L":
                state = State(eos, temperature=sd["temperature"]*KELVIN, pressure=sd["pressure"]*PASCAL, density_initialization="liquid")
                #print("liq")
            if sd["state"] == "G":
                state = State(eos, temperature=sd["temperature"]*KELVIN, pressure=sd["pressure"]*PASCAL, density_initialization="vapor")
                #print("vap")
            else:
                state = State(eos, temperature=sd["temperature"]*KELVIN, pressure=sd["pressure"]*PASCAL)  
            sd["rho"] = state.partial_density[0] / MOL_m3
        else:
            state = State(eos, temperature=sd["temperature"]*KELVIN, density=sd["rho"]*(MOL/METER**3) )
            sd["pressure"] = state.pressure() / PASCAL

        sd["s_total"] = state.specific_entropy(Contributions.Total) *M / J_molK
        sd["s_res"] = state.specific_entropy(Contributions.ResidualNvt) *M / J_molK
        sd["s_res*"] = sd["s_res"]/ KB /NAV
        sd["s_res**"] = sd["s_res*"]/m
        
        sd["E_res"] = state.specific_internal_energy(Contributions.ResidualNvt)*M / J_mol
        sd["H_res"] = state.specific_enthalpy(Contributions.ResidualNvt)*M / J_mol
        sd["G_res"] = state.specific_gibbs_energy(Contributions.ResidualNvt)*M / J_mol
        sd["A_res"] = state.specific_helmholtz_energy(Contributions.ResidualNvt)*M / J_mol
        sd["V"] = 1/sd["rho"]
        
        sd["c_p"] = state.c_p() / JOULE *MOL*KELVIN
        sd["c_v"] = state.c_v() / JOULE *MOL*KELVIN
        
        vle = PhaseDiagram.pure(eos,min_temperature=sd["temperature"]*KELVIN,npoints=100)
        vle_state_vapor = State(eos, temperature=vle.vapor.temperature[0], density=vle.vapor.density[0])
        vle_state_liquid = State(eos, temperature=vle.liquid.temperature[0], density=vle.liquid.density[0])
        sd["s_res_vle_gas"] = vle_state_vapor.specific_entropy(Contributions.ResidualNvt)*M  / J_molK
        sd["s_res_vle_liq"] = vle_state_liquid.specific_entropy(Contributions.ResidualNvt)*M  / J_molK
        sd["s_vap"] = vle_state_liquid.specific_entropy(Contributions.ResidualNvt)*M - vle_state_vapor.specific_entropy(Contributions.ResidualNvt)*M  
        sd["s_vap"] = sd["s_vap"]/ J_molK
        
        sd["dE_vap"] = vle_state_liquid.specific_internal_energy(Contributions.ResidualNvt)*M - vle_state_vapor.specific_internal_energy(Contributions.ResidualNvt)*M 
        sd["dE_vap"] = sd["dE_vap"]/ J_mol
        
        sd["dH_vap"] = vle_state_liquid.specific_enthalpy(Contributions.ResidualNvt)*M - vle_state_vapor.specific_enthalpy(Contributions.ResidualNvt)*M 
        sd["dH_vap"] = sd["dH_vap"]/ J_mol
        
        sd["rho_vap_liq"] = vle_state_liquid.partial_density[0] / MOL_m3
        sd["V_vap_liq"] = 1/sd["rho_vap_liq"]
        
        sd["rho_vap_gas"] = vle_state_vapor.partial_density[0] / MOL_m3
        sd["V_vap_gas"] = 1/sd["rho_vap_gas"]
        sd["dV_vap"] = sd["V_vap_liq"] - sd["V_vap_gas"]
        sd["dpressure"] = (vle_state_liquid.pressure() - vle_state_vapor.pressure())/PASCAL
        dummy = get_viscosity_CE(sd["temperature"]*KELVIN,parameters)
        sd["eta_CE"] = dummy[0] / PS
        sd["eta_CE_ig"] = dummy[1] / PS
        sd["eta_CE_m"] = dummy[2] / PS
        sd["omega22"] = dummy[3]
        sd["M"] = M /(KILO*GRAM/MOL)
        sd["m"] = m
        sd["sigma"] = sigma/METER 
        sd["epsilon"] = epsilon/KELVIN
        sd["R"] = RGAS / (JOULE/MOL/KELVIN)

        sd["dp_drho_npt"] = state.dp_drho(Contributions.ResidualNpt) /JOULE*MOL
        sd["dp_drho_nvt"] = state.dp_drho(Contributions.ResidualNvt) /JOULE*MOL
        sd["dp_drho_tot"] = state.dp_drho(Contributions.Total) /JOULE*MOL
        sd["speed_of_sound"] = state.speed_of_sound() /METER *SECOND

        sd["success"] = 1
    except:
        sd["success"] = 0
    return sd


def get_lamda(T,parameter):
    epsilon = parameters.pure_records[0].model_record.epsilon_k    
    sigma = parameters.pure_records[0].model_record.sigma

    omega22 = get_omega22(T/epsilon)
    llambda = np.sqrt( np.sqrt(2)*np.pi*omega22  )*sigma /1e10
    return llambda


def get_VN(V):
    return (V/6.02214076e+23)**(1/3)

def get_V_eqd(T,parameters,pp):
    llambda1 = get_lamda(T,parameters)
    return llambda1**3*6.02214076e+23*pp**(3/2)
