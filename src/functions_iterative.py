import xpress as xp
import numpy as np
from scipy.interpolate import interp1d
from time import time
import subprocess
from configparser import ConfigParser
import re

xp.setOutputEnabled(False)


class Basis:
    """ Class to store basis with Xpress """

    def __init__(self, rstatus:list=[], cstatus:list=[]) -> None:
        """
        Create a new basis.

        Parameters
        ----------
        rstatus:list : 
            Row basis obtained with problem.getbasis() (Default value = [])
        cstatus:list :
            Column basis obtained with problem.getbasis() (Default value = [])

        Returns
        -------
        None

        """
        self.rstatus = rstatus
        self.cstatus = cstatus

    def not_empty(self) -> bool:
        """ Check if a basis isn't empty (True) or not (False)"""
        return(len(self.rstatus)!=0)
class AntaresParameter:
    """ Class to store time-related parameters of an Antares study"""

    def __init__(self, S:int=52, H:int=168, NTrain:int=1) -> None:
        """
        Create a new set of parameters.

        Parameters
        ----------
        S:int :
            Total number of weeks in a year (Default value = 52)
        H:int :
            Total number of hours in a week (Default value = 168)
        NTrain:int :
            Total number of Monte Carlo years, ie scenarios (Default value = 1)

        Returns
        -------
        None
        """
        self.S = S
        self.H = H
        self.NTrain = NTrain
        self.list_problem = [[[] for k in range(NTrain)] for s in range(S)]

    def get_S(self) -> int:
        """ Return total number of weeks. """
        return(self.S)
    
    def get_H(self) -> int:
        """ Return total number of hours in a week. """
        return(self.H)
    
    def get_NTrain(self) -> int:
        """ Return total number of scenarios. """
        return(self.NTrain)

class Reservoir:
    """Describes reservoir parameters"""

    def __init__(self, param:AntaresParameter, dir_study:str, name_area:str, final_level:bool=True) -> None:
        """
        Create a new reservoir. 

        Parameters
        ----------
        param:AntaresParameter : 
            Time-related parameters
        capacity:float :
            Capacity of the reservoir (in MWh)
        efficiency:float :
            Efficiency of the pumping (between 0 and 1)
        dir_study:str :
            Path to the Antares study
        name_area:str :
            Name of the area where is located the reservoir
        name:str :
            Name of the reservoir
        final_level:bool :
            True if final level should be egal to initial level (Default value = True)

        Returns
        -------
        None
        """

        H = param.get_H()

        hydro_ini = ConfigParser()
        hydro_ini.read(dir_study+"/input/hydro/hydro.ini")
        
        self.capacity = hydro_ini.getfloat("reservoir capacity",name_area)

        courbes_guides = np.loadtxt(dir_study+"/input/hydro/common/capacity/reservoir_"+name_area+".txt")[:,[0,2]]*self.capacity
        assert(courbes_guides[0,0]==courbes_guides[0,1])
        self.initial_level = courbes_guides[0,0]
        Xmin = courbes_guides[6:365:7,0]
        Xmax = courbes_guides[6:365:7,1]
        self.Xmin = np.concatenate((Xmin,Xmin[[0]]))
        self.Xmax = np.concatenate((Xmax,Xmax[[0]]))
        if final_level:
            self.Xmin[51] = self.initial_level
            self.Xmax[51] = self.initial_level
        

        self.inflow = np.loadtxt(dir_study+"/input/hydro/series/"+name_area+"/mod.txt")[6:365:7]*7/H 
        self.area = name_area

        P_turb = np.loadtxt(dir_study+"/input/hydro/common/capacity/maxpower_"+name_area+".txt")[:,0]
        P_pump = np.loadtxt(dir_study+"/input/hydro/common/capacity/maxpower_"+name_area+".txt")[:,2]
        self.P_turb = P_turb
        self.P_pump = P_pump
        self.efficiency = hydro_ini.getfloat("pumping efficiency",name_area)

def generate_mps_file(study_path:str,antares_path:str) -> str :
    name_solver = antares_path.split("/")[-1]
    assert("solver" in name_solver)
    assert(float(name_solver.split("-")[1])>=8.7)
    res = subprocess.run([antares_path, "--named-mps-problems","--name=export_mps",study_path],capture_output=True, text=True)
    assert("Quitting the solver gracefully" in res.stdout)
    output = res.stdout.split("\n")
    idx_line = [l for l in output if " Output folder : " in l]
    assert(len(idx_line)>=1)
    output_folder = idx_line[0].split(" Output folder : ")[1]
    output_folder = output_folder.replace("\\","/")
    return(output_folder)
class AntaresProblem :
    """ Class to store an Xpress optimization problem describing the problem solved by Antares for one week and one scenario. """

    def __init__(self,year:int,week:int,path:str,itr:int=1) -> None:
        """
        Create a new Xpress problem and load the problem stored in the associated mps file.

        Parameters
        ----------
        year:int :
            Scenario considered
        week:int :
            Week considered
        path:str :
            Path where mps files are stored
        itr:int :
            Antares iteration considered (Default value = 1)

        Returns
        -------
        None
        """
        self.year = year
        self.week = week
        self.path = path
        
        model = xp.problem()
        model.controls.outputlog = 0
        model.controls.threads = 1
        model.controls.scaling = 0
        model.controls.presolve = 0
        model.controls.feastol = 1.e-7
        model.controls.optimalitytol = 1.e-7
        model.controls.xslp_log = -1
        model.controls.lplogstyle = 0
        model.read(path+f"/problem-{year+1}-{week+1}--optim-nb-{itr}.mps")
        self.model = model

        self.basis = []
        self.control_basis = []

    def add_basis(self, basis:Basis, control_basis:float) -> None:
        """
        Store a new basis for the optimization problem.

        Parameters
        ----------
        basis:Basis :
            New basis to store
        control_basis:float :
            Reservoir control for which the problem has been solved

        Returns
        -------
        None
        """
        self.basis.append(basis)
        self.control_basis.append(control_basis)

    def find_closest_basis(self, control:float) -> None:
        """
        Among stored basis, return the closest one to the given control.

        Parameters
        ----------
        control:float :
            Control for which we want to solve the optimization problem

        Returns
        -------

        """
        u = np.argmin(np.abs(np.array(self.control_basis)-control))
        return(self.basis[u])

    def create_weekly_problem_itr(self, param:AntaresParameter, reservoir:Reservoir, pen_low:float=0, pen_high:float=0,pen_final:float=0)-> None :
        """
        Modify the Xpress problem to take into account reservoir constraints and manage reservoir with Bellman values and penalties on rule curves.

        Parameters
        ----------
        param:AntaresParameter :
            Time-related parameters
        reservoir:Reservoir :
            Considered reservoir
        pen_low:float :
            Penalty for violating bottom rule curve (Default value = 0)
        pen_high:float :
            Penalty for violating top rule curve (Default value = 0)
        pen_final:float :
            Penalty for violating rule curves at the end of the year (Default value = 0)

        Returns
        -------
        None
        """

        S = param.get_S()
        H = param.get_H()

        model = self.model
        
        self.delete_variable(H=H,name_variable=f'^HydroLevel::area<{reservoir.area}>::hour<.')
        self.delete_variable(H=H,name_variable=f'^Overflow::area<{reservoir.area}>::hour<.')
        self.delete_constraint(H=H,name_constraint=f'^AreaHydroLevel::area<{reservoir.area}>::hour<.')

        cst = model.getConstraint()
        binding_id = [i for i in range(len(cst)) if re.search(f'^HydroPower::area<{reservoir.area}>::week<.',cst[i].name)]
        assert len(binding_id)==1

        x_s = xp.var("x_s",lb = 0, ub = reservoir.capacity)
        model.addVariable (x_s)          # State at the begining of the current week

        x_s_1 = xp.var("x_s_1",lb = 0, ub = reservoir.capacity)
        model.addVariable (x_s_1) # State at the begining of the following week

        U = xp.var("u",lb = -reservoir.P_pump[7*self.week]*reservoir.efficiency*H, ub = reservoir.P_turb[7*self.week]*H)
        model.addVariable (U) # State at the begining of the following week

        model.addConstraint(x_s_1 <= x_s - U + reservoir.inflow[self.week,self.year]*H)

        y = xp.var("y")

        model.addVariable (y)    # Penality for violating guide curves

        if self.week!=S-1:
            model.addConstraint(y >=  -pen_low* (x_s_1 - reservoir.Xmin[self.week]))
            model.addConstraint(y >=  pen_high* (x_s_1 - reservoir.Xmax[self.week]))
        else :
            model.addConstraint(y >=  -pen_final* (x_s_1 - reservoir.Xmin[self.week]))
            model.addConstraint(y >=  pen_final* (x_s_1 - reservoir.Xmax[self.week]))

        z = xp.var("z",lb = float('-inf'), ub =  float('inf'))

        model.addVariable (z) # Auxiliar variable to introduce the piecewise representation of the future cost

        self.binding_id = binding_id
        self.U = U
        self.x_s = x_s
        self.x_s_1 = x_s_1
        self.z = z
        self.y = y

    def delete_variable(self,H,name_variable):
        model = self.model
        var = model.getVariable()
        var_id = [i for i in range(len(var)) if re.search(name_variable,var[i].name)]
        assert len(var_id) in [0,H]
        if len(var_id)==H:
            model.delVariable(var_id)

    def delete_constraint(self,H,name_constraint):
        model = self.model
        cons = model.getConstraint()
        cons_id = [i for i in range(len(cons)) if re.search(name_constraint,cons[i].name)]
        assert len(cons_id) in [0,H]
        if len(cons_id)==H:
            model.delConstraint(cons_id)


    def modify_weekly_problem_itr(self,control:float,i:int, prev_basis:Basis=None) -> tuple[float, float, int, Basis, float]:
        """
        Modify and solve problem to evaluate weekly cost associated with a particular control of the reservoir.

        Parameters
        ----------
        control:float :
            Control to evaluate
        i:int :
            Iteration of the iterative algorithm
        prev_basis:Basis :
            Basis used at a previous resolution of a similar problem (Default value = None)

        Returns
        -------
        beta:float :
            Total cost
        lamb:float :
            Dual value associated to the control constraint
        itr:int :
            Total number of simplex iterations used to solve the problem
        prev_basis:Basis : 
            Basis output by the resolution
        t:float :
            Time spent solving the problem
        """

        if (prev_basis.not_empty())&(i==0):
            self.model.loadbasis(prev_basis.rstatus, prev_basis.cstatus)

        if i>=1:
            basis = self.find_closest_basis(control=control)
            self.model.loadbasis(basis.rstatus,basis.cstatus)

        rbas = []
        cbas = []

        self.model.chgrhs(self.binding_id,[control])
        debut_1 = time()
        self.model.lpoptimize()
        fin_1 = time()
        

        if self.model.attributes.lpstatus==1:
            beta = self.model.getObjVal()
            lamb = self.model.getDual(self.binding_id)[0]
            itr = self.model.attributes.SIMPLEXITER

            
            self.model.getbasis(rbas,cbas)
            self.add_basis(basis=Basis(rbas,cbas), control_basis=control)
            
            if i==0:
                prev_basis.rstatus = rbas 
                prev_basis.cstatus = cbas
            return(beta,lamb,itr, prev_basis, fin_1-debut_1)
        else :
            
            raise(ValueError)
        
class RewardApproximation:
    """ Class to store and update reward approximation for a given week and a given scenario"""

    def __init__(self, lb_control:float, ub_control:float, ub_reward:float) -> None:
        """
        Create a new reward approximation

        Parameters
        ----------
        lb_control:float :
            Lower possible bound on control
        ub_control:float :
            Upper possible bound on control
        ub_reward:float :
            Upper bound on reward

        Returns
        -------
        None
        """
        self.breaking_point = [lb_control, ub_control]
        self.list_cut = [(0, ub_reward)]

    def reward_function(self) -> callable:
        """ Return a function to evaluate reward at any point based on the current approximation. """
        return(lambda x: min([cut[0]*x+cut[1] for cut in self.list_cut]))
    
    def update_reward_approximation(self,lamb:float,beta:float,new_control:float) -> None:
        """
        Update reward approximation by adding a new cut

        Parameters
        ----------
        lamb:float :
            Total cost that defined reward at the given control
        beta:float :
            Dual value associated with the control constraint, gives the slope of reward
        new_control:float :
            Control evaluated

        Returns
        -------
        None
        """

        Gs = self.reward_function()
        new_cut = lambda x:-lamb*x -beta + lamb*new_control
        new_reward = []
        new_points = [self.breaking_point[0]]

        if (len(self.breaking_point)!=len(self.list_cut)+1) :
            raise(ValueError)
        
        for i in range(len(self.breaking_point)):
            if i==0:
                if (new_cut(self.breaking_point[i]) < Gs(self.breaking_point[i])) :
                    new_reward.append((-lamb,-beta + lamb*new_control))
                    if (new_cut(self.breaking_point[i+1]) >= Gs(self.breaking_point[i+1])) :
                        if -lamb-self.list_cut[i][0]!=0:
                            new_points.append(-(-beta + lamb*new_control-self.list_cut[i][1])/(-lamb-self.list_cut[i][0]))
                            new_reward.append(self.list_cut[i])
                elif (new_cut(self.breaking_point[i]) >= Gs(self.breaking_point[i])):
                    new_reward.append(self.list_cut[i])
                    if (new_cut(self.breaking_point[i+1]) < Gs(self.breaking_point[i+1])) :
                        if -lamb-self.list_cut[i][0]!=0:
                            new_points.append(-(-beta + lamb*new_control-self.list_cut[i][1])/(-lamb-self.list_cut[i][0]))
                            new_reward.append((-lamb,-beta + lamb*new_control))
            elif i==len(self.breaking_point)-1:
                new_points.append(self.breaking_point[-1])
            else :
                if (new_cut(self.breaking_point[i]) >= Gs(self.breaking_point[i])) :
                    new_reward.append(self.list_cut[i])
                    new_points.append(self.breaking_point[i])
                    if (new_cut(self.breaking_point[i+1]) < Gs(self.breaking_point[i+1])) :
                        if -lamb-self.list_cut[i][0]!=0:
                            new_reward.append((-lamb,-beta + lamb*new_control))
                            new_points.append(-(-beta + lamb*new_control-self.list_cut[i][1])/(-lamb-self.list_cut[i][0]))
                elif (new_cut(self.breaking_point[i]) < Gs(self.breaking_point[i])) and (new_cut(self.breaking_point[i+1]) >= Gs(self.breaking_point[i+1])) :
                    if -lamb-self.list_cut[i][0]!=0:
                        new_reward.append(self.list_cut[i])
                        new_points.append(-(-beta + lamb*new_control-self.list_cut[i][1])/(-lamb-self.list_cut[i][0]))
                    
        self.breaking_point = new_points
        self.list_cut = new_reward
        
def get_penalty(s:int,S:int,reservoir:Reservoir,pen_final:float,pen_low:float,pen_high:float) -> callable:
    """ 
    Return a function to evaluate penalities for violating rule curves for any level of stock.

    Parameters
    ----------
    s:int :
        Week considered
    S:int :
        Total number of weeks
    reservoir:Reservoir :
        Reservoir considered
    pen_final:float :
        Penalty for violating final rule curves
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve

    Returns
    -------

    """
    if s==S-1:
        pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_final*(reservoir.Xmin[s]),0,0,-pen_final*(reservoir.capacity-reservoir.Xmax[s])])
    else :
        pen = interp1d([0,reservoir.Xmin[s],reservoir.Xmax[s],reservoir.capacity],[-pen_low*(reservoir.Xmin[s]),0,0,-pen_high*(reservoir.capacity-reservoir.Xmax[s])])
    return(pen)

def solve_weekly_problem_with_approximation(points:list, X:np.array, inflow:float, lb:float, ub:float, level_i:float, xmax:float, xmin:float, cap:float, pen:callable, V_fut:callable, Gs:callable) -> tuple[float, float, float]:
    """
    Optimize control of reservoir during a week based on reward approximation and current Bellman values.

    Parameters
    ----------
    points:list :
        Breaking points in reward approximation
    X:np.array :
        Breaking points in Bellman values approximation
    inflow:float :
        Inflow in the reservoir during the week
    lb:float :
        Lower possible bound on control
    ub:float :
        Upper possible bound on control
    level_i:float :
        Initial level of reservoir at the beginning of the week
    xmax:float :
        Upper rule curve at the end of the week
    xmin:float :
        Bottom rule curve an the end of the week
    cap:float :
        Capacity of the reservoir
    pen:callable :
        Penalties for violating rule curves at the end of the week
    V_fut:callable :
        Bellman values at the end of the week
    Gs:callable :
        Reward approximation for the current week

    Returns
    -------
    Vu:float : 
        Optimal objective value
    xf:float : 
        Final level of sotck
    control:float : 
        Optimal control
    """
    Vu = float('-inf')

    for i_fut in range(len(X)):
        u = -X[i_fut] + level_i + inflow
        if lb <= u <= ub:
            G = Gs(u)
            penalty = pen(X[i_fut])
            if (G + V_fut(X[i_fut])+penalty) > Vu:
                Vu = G + V_fut(X[i_fut])+penalty
                xf = X[i_fut]
                control = u
                
    for u in range(len(points)):
        state_fut = min(cap,level_i - points[u] + inflow) 
        if 0 <= state_fut :
            penalty = pen(state_fut)
            G = Gs(points[u])
            if (G + V_fut(state_fut)+penalty) > Vu:
                Vu = (G + V_fut(state_fut)+penalty)
                xf = state_fut
                control = points[u]

    Umin = level_i+ inflow-xmin
    if lb <= Umin <= ub:
        state_fut = level_i - Umin + inflow
        penalty = pen(state_fut)
        if (Gs(Umin) + V_fut(state_fut)+penalty) > Vu:
            Vu = Gs(Umin) + V_fut(state_fut)+penalty
            xf = state_fut
            control = Umin

    Umax = level_i+ inflow-xmax
    if lb <= Umax <= ub:
        state_fut = level_i - Umax + inflow 
        penalty = pen(state_fut)
        if (Gs(Umax) + V_fut(state_fut)+penalty) > Vu:
            Vu = Gs(Umax) + V_fut(state_fut)+penalty
            xf = state_fut
            control = Umax

    control = min(-(xf-level_i-inflow), ub)
    return (Vu, xf, control)

def calculate_VU(param:AntaresParameter,reward:list[list[RewardApproximation]], reservoir:Reservoir,X:np.array, pen_low:float, pen_high:float, pen_final:float) -> np.array:
    """
    Calculate Bellman values for every week based on reward approximation

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    reward:list[list[RewardApproximation]] :
        Reward approximation for every week and every scenario
    reservoir:Reservoir :
        Reservoir considered
    X:np.array :
        Discretization of stock levels
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    pen_final:float :
        Penalty for violating final rule curves

    Returns
    -------

    """

    S = param.get_S()
    NTrain = param.get_NTrain()
    H = param.get_H()

    V = np.zeros((len(X), S+1, NTrain))

    for s in range(S-1,-1,-1):
        
        pen = get_penalty(s=s,S=S,reservoir=reservoir,pen_final=pen_final,pen_low=pen_low, pen_high=pen_high)

        for k in range(NTrain):
            V_fut = interp1d(X, V[:, s+1,k])
            Gs = reward[s][k].reward_function()
            for i in range(len(X)):
                
                Vu, _, _ = solve_weekly_problem_with_approximation(points=reward[s][k].breaking_point, X=X, inflow=reservoir.inflow[s,k]*H, lb=-reservoir.P_pump[7*s]*H, ub=reservoir.P_turb[7*s]*H,level_i=X[i], xmax=reservoir.Xmax[s], xmin=reservoir.Xmin[s], cap=reservoir.capacity,pen=pen, V_fut=V_fut, Gs=Gs)
            
                V[i, s, k] = Vu + V[i,s,k]

        V[:,s,:] = np.repeat(np.mean(V[:,s,:],axis=1,keepdims=True),NTrain,axis=1)
    return np.mean(V,axis=2)

def compute_x_multi_scenario(param:AntaresParameter,reservoir:Reservoir,X:np.array,V:np.array,reward:list[list[RewardApproximation]],pen_low:float,pen_high:float, pen_final:float, itr:int) -> tuple[np.array, np.array]:
    """
    Compute several optimal trajectories for the level of stock based on reward approximation and Bellman values. The number of trajectories is equal to the number of scenarios but trajectories doesn't depend on Monte Carlo years, ie for a given trajectory each week correspond to a random scenario.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    reservoir:Reservoir :
        Reservoir considered
    X:np.array :
        Discretization of stock levels
    V:np.array :
        Bellman values
    reward:list[list[RewardApproximation]] :
        Reward approximation for every week and every scenario
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    pen_final:float :
        Penalty for violating final rule curves
    itr:int :
        Iteration of iterative algorithm

    Returns
    -------
    initial_x:np.array :
        Trajectories
    controls:np.array :
        Controls associated to trajectories
    """

    S = param.get_S()
    NTrain = param.get_NTrain()
    H = param.get_H()

    initial_x = np.zeros((S+1,NTrain))
    initial_x[0] = reservoir.initial_level
    np.random.seed(19*itr)
    controls = np.zeros((S,NTrain))
    
    for s in range(S):
        
        V_fut = interp1d(X, V[:, s+1])
        for j,k_s in enumerate(np.random.permutation(range(NTrain))):
        
            pen = get_penalty(s=s,S=S,reservoir=reservoir,pen_final=pen_final,pen_low=pen_low, pen_high=pen_high)
            Gs = reward[s][k_s].reward_function()

            _, xf, u = solve_weekly_problem_with_approximation(points=reward[s][k_s].breaking_point, X=X, inflow=reservoir.inflow[s,k_s]*H, lb=-reservoir.P_pump[7*s]*H, ub=reservoir.P_turb[7*s]*H,level_i=initial_x[s,j], xmax=reservoir.Xmax[s], xmin=reservoir.Xmin[s], cap=reservoir.capacity,pen=pen, V_fut=V_fut, Gs=Gs)

            initial_x[s+1,j] = xf
            controls[s,k_s] = u

    return(initial_x, controls)

def find_likely_control(param:AntaresParameter,reservoir:Reservoir,X:np.array,V:np.array,reward:list[list[RewardApproximation]],pen_low:float,pen_high:float, pen_final:float, level_i:float, s:int, k:int) -> float:
    """
    Compute a control which is likely to be optimal in the real optimization problem based on reward approximation.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    reservoir:Reservoir :
        Reservoir considered
    X:np.array :
        Discretization of Bellman values
    V:np.array :
        Bellman values
    reward:list[list[RewardApproximation]] :
        Reward approximation for every week and every scenario
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    pen_final:float :
        Penalty for violating final rule curves
    level_i:float :
        Initial level of stock
    s:int :
        Week considered
    k:int :
        Scenario considered

    Returns
    -------
    Likely control
    """

    S = param.get_S()
    H = param.get_H()
    
    V_fut = interp1d(X, V[:, s+1])
        
        
    pen = get_penalty(s=s,S=S,reservoir=reservoir,pen_final=pen_final,pen_low=pen_low, pen_high=pen_high)
    Gs = reward[s][k].reward_function()

    _, _, u = solve_weekly_problem_with_approximation(points=reward[s][k].breaking_point, X=X, inflow=reservoir.inflow[s,k]*H, lb=-reservoir.P_pump[7*s]*H, ub=reservoir.P_turb[7*s]*H,level_i=level_i, xmax=reservoir.Xmax[s], xmin=reservoir.Xmin[s], cap=reservoir.capacity,pen=pen, V_fut=V_fut, Gs=Gs)

    return(u)

def compute_upper_bound(param:AntaresParameter,reservoir:Reservoir,list_models:list[list[AntaresProblem]],X:np.array,V:np.array,G:list[list[RewardApproximation]],pen_low:float,pen_high:float,pen_final:float) -> tuple[float, np.array,np.array]:
    """
    Compute an approximate upper bound on the overall problem by solving the real complete Antares problem with Bellman values.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters
    reservoir:Reservoir :
        Reservoir considered
    list_models:list[list[AntaresProblem]] :
        Optimization problems for every week and every scenario
    X:np.array :
        Discretization of Bellman values
    V:np.array :
        Bellman values
    G:list[list[RewardApproximation]] :
        Reward approximation for every week and every scenario
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    pen_final:float :
        Penalty for violating final rule curves

    Returns
    -------
    upper_bound:float : 
        Upper bound on the overall problem
    controls:np.array :
        Optimal controls for every week and every scenario
    current_itr:np.array :
        Time and simplex iterations used to solve the problem
    """
    
    S = param.get_S()
    H = param.get_H()
    NTrain = param.get_NTrain()

    current_itr = np.zeros((S,NTrain,2))
    
    cout = 0
    controls = np.zeros((S,NTrain))
    for k in range(NTrain):
        
        level_i = reservoir.initial_level
        for s in range(S):
            print(f"{k} {s}",end="\r")
            m = list_models[s][k]

            nb_cons = m.model.attributes.rows

            m.model.chgmcoef(m.binding_id,[m.U],[-1])
            m.model.chgrhs(m.binding_id,[0])

            m.model.chgobj([m.y,m.z], [1,1])

            if len(m.control_basis)>=1:
                if len(m.control_basis)>=2:
                    likely_control = find_likely_control(param=param,reservoir=reservoir,X=X,V=V,reward=G,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final, level_i=level_i, s=s, k=k)
                else :
                    likely_control = 0     
                basis = m.find_closest_basis(likely_control)
                m.model.loadbasis(basis.rstatus,basis.cstatus)
        
            for j in range(len(X)-1):
                if (V[j+1, s+1]<float('inf'))&(V[j, s+1]<float('inf')):
                    m.model.addConstraint(m.z >= (-V[j+1, s+1] + V[j, s+1]) / (X[j+1] - X[j]) * (m.x_s_1 - X[j]) - V[j, s+1])

            cst_initial_level = m.x_s == level_i
            m.model.addConstraint(cst_initial_level)

            rbas = []
            cbas = []

            debut_1 = time()
            m.model.lpoptimize()
            fin_1 = time()

            if m.model.attributes.lpstatus==1:

                m.model.getbasis(rbas,cbas)
                m.add_basis(basis=Basis(rbas[:nb_cons],cbas),control_basis=m.model.getSolution(m.U))

                beta = m.model.getObjVal()
                xf = m.model.getSolution(m.x_s_1)
                z = m.model.getSolution(m.z)
                y = m.model.getSolution(m.y)
                m.model.delConstraint(range(nb_cons,m.model.attributes.rows))
                m.model.chgmcoef(m.binding_id,[m.U],[0])
                
                m.model.chgobj([m.y,m.z], [0,0])
                cout += beta
                controls[s,k]=-(xf-level_i-reservoir.inflow[s,k]*H)
                level_i = xf
                if s!=S-1:
                    cout += - z - y

                itr = m.model.attributes.SIMPLEXITER

            else :
                raise(ValueError)
            current_itr[s,k] = (itr,fin_1-debut_1)

        upper_bound = cout/NTrain
    return(upper_bound, controls, current_itr)

def calculate_reward(param:AntaresParameter,controls:list[list[float]], list_models:list[list[AntaresProblem]], G:list[list[RewardApproximation]], i:int) -> tuple[np.array, list[list[RewardApproximation]]]:
    """
    Evaluate reward for a set of given controls for each week and each scenario to update reward approximation.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters 
    controls:list[list[float]] :
        Set of controls to evaluate
    list_models:list[list[AntaresProblem]] :
        Optimization problems for every week and every scenario
    G:list[list[RewardApproximation]] :
        Reward approximation to update for every week and every scenario
    i:int :
        Iteration of iterative algorithm

    Returns
    -------
    current_itr:np.array :
        Time and simplex iterations used to solve the problem
    G:list[list[RewardApproximation]] :
        Updated reward approximation
    """

    S = param.get_S()
    NTrain = param.get_NTrain()

    current_itr = np.zeros((S,NTrain,2))
    
    for k in range(NTrain):
        basis_0 = Basis([],[])   
        for s in range(S):
            print(f"{k} {s}",end="\r")

            beta, lamb, itr, basis_0, computation_time = list_models[s][k].modify_weekly_problem_itr(control=controls[s][k],i=i, prev_basis=basis_0)

            G[s][k].update_reward_approximation(lamb,beta,controls[s][k])
            
            current_itr[s,k] = (itr,computation_time)

    return(current_itr,G)

def itr_control(param:AntaresParameter, reservoir:Reservoir, output_path:str, pen_low:float, pen_high:float,X:np.array, N:int, pen_final:float, tol_gap:float) -> tuple[np.array, list[list[RewardApproximation]], np.array, list[float], list[np.array], list[np.array]]:
    """
    Algorithm to evaluate Bellman values. Each iteration of the algorithm consists in computing optimal trajectories based on reward approximation then evaluating rewards for those trajectories and finally updating reward approximation and calculating Bellman values. The algorithm stops when a certain number of iteratin is done or when the gap between the lower bound and the upper bound is small enough.

    Parameters
    ----------
    param:AntaresParameter :
        Time-related parameters for the Antares study
    reservoir:Reservoir :
        Reservoir considered for Bellman values
    output_path:str :
        Path to mps files describing optimization problems
    pen_low:float :
        Penalty for violating bottom rule curve
    pen_high:float :
        Penalty for violating top rule curve
    X:np.array :
        Discretization of sotck levels for Bellman values
    N:int :
        Maximum number of iteration to do
    pen_final:float :
        Penalty for violating final rule curves
    tol_gap:float :
        Relative tolerance gap for the termination of the algorithm 

    Returns
    -------
    V:np.array :
        Bellman values
    G:list[list[RewardApproximation]] :
        Reward approximation
    itr:np.array :
        Time and simplex iterations used to solve optimization problems at each iteration
    tot_t:list[float] :
        Time spent at each iteration
    controls_upper:list[np.array] :
        Optimal controls found at each iteration during the evaluation of the upper bound
    traj:list[np.array] :
        Trajectories computed at each iteration
    """

    S = param.get_S()
    NTrain = param.get_NTrain()
    H = param.get_H()

    tot_t = []
    debut = time()
    
    list_models = [[] for i in range(S)]
    for s in range(S):
        for k in range(NTrain):
            m = AntaresProblem(year=k,week=s,path=output_path,itr=1)
            m.create_weekly_problem_itr(param=param,reservoir=reservoir,pen_low=pen_low,pen_high=pen_high,pen_final=pen_final)
            list_models[s].append(m)
    
    V = np.zeros((len(X), S+1))
    G = [[RewardApproximation(lb_control=-reservoir.P_pump[7*s]*H,ub_control=reservoir.P_turb[7*s]*H,ub_reward=0) for k in range(NTrain)] for s in range(S)]
    
    itr_tot = []
    controls_upper = []
    traj = []

    i = 0
    gap = 1e3
    fin = time()
    tot_t.append(fin-debut)
    while (gap>=tol_gap and gap>=0) and i <N : #and (i<3):
        debut = time()

        initial_x, controls = compute_x_multi_scenario(param=param,reservoir=reservoir,X=X,V=V,reward=G,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final, itr=i)
        traj.append(np.array(initial_x))

        current_itr,G = calculate_reward(param=param,controls=controls, list_models=list_models, G=G, i=i)
        itr_tot.append(current_itr)

        V = calculate_VU(param=param,reward=G,reservoir=reservoir,X=X,pen_low=pen_low,pen_high=pen_high, pen_final=pen_final)
        V_fut = interp1d(X, V[:, 0])
        V0 = V_fut(reservoir.initial_level)
            
        upper_bound, controls, current_itr = compute_upper_bound(param=param,reservoir=reservoir,list_models=list_models,X=X,V=V,G=G,pen_low=pen_low,pen_high=pen_high,pen_final=pen_final)
        itr_tot.append(current_itr)      
        controls_upper.append(controls)

        gap = upper_bound+V0
        print(gap, upper_bound,-V0)
        gap = gap/-V0
        i+=1
        fin = time()
        tot_t.append(fin-debut)
    return (V, G, np.array(itr_tot),tot_t, controls_upper, traj)