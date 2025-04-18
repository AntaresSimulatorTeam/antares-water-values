from usage_values import UsageValuesTempo
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class TrajectoriesTempo :
    def __init__(self,
                usage_values:UsageValuesTempo,
                trajectories_red:Optional[np.ndarray]=None):
        
        self.usage_values=usage_values
        self.residual_load=self.usage_values.bellman_values.gain_function.residual_load
        self.capacity=self.usage_values.capacity
        self.nb_week=self.usage_values.nb_week
        self.nb_scenarios=self.usage_values.bellman_values.nb_scenarios
        self.start_week=self.usage_values.start_week
        self.end_week=self.usage_values.end_week
        self.max_control=self.usage_values.bellman_values.max_control

        self.trajectories=np.zeros((self.nb_scenarios,self.nb_week-1)) #trajectoire de tirage des jours tempo, trajectoire[i] est le tirage de jours
        # tempo pour la semaine i et stock_i+1 = stock_i - trajectoire[i], car le stock est calculé en début de semaine
        self.stock=np.zeros((self.nb_scenarios,self.nb_week))
        
        self.trajectories_red=trajectories_red

        self.compute_trajectories()
        self.compute_stock()

        self.compute_white_trajectories()
        self.compute_white_stock()
    
    def compute_trajectories(self)-> None:

        for s in range(self.nb_scenarios):
            remaining_capacity=self.capacity
            for w in range(self.start_week+1,self.end_week):
                if remaining_capacity==0:
                    self.trajectories[s,w-self.start_week-1]=0
                    continue
                
                thresholds=self.usage_values.usage_values[w,:remaining_capacity]
                week_start=(w-1)*7+2
                week_end=week_start+7
                week_days=self.residual_load[week_start*24:week_end*24,s].reshape(7,24).sum(axis=1)
                sorted_load=(np.sort(week_days[:self.max_control]))[::-1]

                control = 0
                for d in range(min(self.max_control, remaining_capacity)):
                    idx = remaining_capacity - 1 - d
                    if sorted_load[d] < thresholds[idx]: 
                        break         # puis on s’arrête
                    control += 1      # sinon on continue

                if self.trajectories_red is not None:
                    if w < 18:
                        # On protège les jours rouges à venir
                        if remaining_capacity - control < 22:
                            control = int(max(0, remaining_capacity - 22))

                    elif 18 <= w < 40:
                        red_weeks_idx = w - 18
                        control_r = self.trajectories_red[s, red_weeks_idx-1]

                        # Imposer que control_wr >= control_r
                        control = int(max(control, control_r))

                        # Ne pas dépasser ce qu'on peut consommer
                        control = int(min(control, remaining_capacity))


                self.trajectories[s, w - self.start_week -1] = control
                remaining_capacity -= control
                remaining_capacity=int(max(remaining_capacity,0))
    
    def compute_white_trajectories(self)->None:
        if self.trajectories_red is not None:
            self.trajectories_white=np.copy(self.trajectories)
            for s in range(self.nb_scenarios):
                for w in range(9,30):
                    index=w
                    red_index=w-9
                    self.trajectories_white[s,index]-=self.trajectories_red[s,red_index]
        else:
            self.trajectories_white=np.array([])

    def compute_stock(self) -> None:
        for s in range(self.nb_scenarios):
            self.stock[s,0]=self.capacity
            remaining_capacity=self.capacity
            for w in range(self.start_week+1,self.end_week):
                self.stock[s,w-self.start_week]=remaining_capacity-self.trajectories[s,w-self.start_week-1]
                remaining_capacity-=self.trajectories[s,w-self.start_week-1]
            
    def compute_white_stock(self)-> None:
        if self.trajectories_red is not None:
            self.stock_white=np.zeros_like(self.stock)
            for s in range(self.nb_scenarios):
                self.stock_white[s,0]=self.capacity
                self.stock_white[s,:self.start_week]=self.stock[s,:self.start_week]-np.full(self.start_week,22)
                for w in range(self.start_week+1,self.end_week):
                    index=w-self.start_week
                    control_w=self.trajectories_white[s,index-1]
                    self.stock_white[s,index]=self.stock_white[s,index-1]-control_w
        else:
            self.stock_white=np.array([])

    def trajectory_for_scenario(self, scenario:int)-> np.ndarray:
        return self.trajectories[scenario]

    def stock_for_scenario(self,scenario:int) -> np.ndarray:
        return self.stock[scenario]
    
    def white_trajectory_for_scenario(self, scenario: int) -> np.ndarray:
        return self.trajectories_white[scenario] if self.trajectories_white[scenario].shape[0]>0 else np.array([])

    def white_stock_for_scenario(self, scenario: int) -> np.ndarray:
        return self.stock_white[scenario] if self.stock_white[scenario].shape[0]>0 else np.array([])