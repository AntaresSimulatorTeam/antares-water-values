from typing import Optional, Union
from inspect import isfunction, isclass

class Caller:
    def __init__(self, **kwargs):
        self.args = kwargs
    
    def update(self, **kwargs):
        self.args = {**self.args, **kwargs}

    def __call__(self, func, returns:Optional[Union[str, tuple[str]]]=None):
        # Extract the function's argument names
        if isfunction(func):
            func_args = func.__code__.co_varnames[:func.__code__.co_argcount]
        elif isclass(func):
            func_args = func.__init__.__code__.co_varnames[:func.__init__.__code__.co_argcount]
        else:
            raise TypeError('Unexpected object passed to Caller')
        
        # Filter the current arguments to match the function's parameters
        relevant_args = {k: self.args[k] for k in func_args if k in self.args}
        
        # Call the function with the filtered arguments and update the helper's args
        result = func(**relevant_args)
        
        if returns is not None:
            if type(returns) == str:
                self.args[returns] = result
            else:
                for ret, res in zip(returns, result):
                    self.args[ret] = res
        return result
    
    def get(self, names:Union[str, tuple[str]]):
        if isinstance(names, str):
            return self.args[names]
        else:
            return tuple([self.args[name] for name in names])