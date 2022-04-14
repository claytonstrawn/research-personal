"""How to use: create a DataCube with a n-d array of floats, and n lists of strings representing the 
values each refers to. Access values as arrays by name from the float list. Recursively applies all dimensions.

Ex:

import numpy as np
from research_personal.data_cube import DataCube

xs = ['a','b']
ys = ['y1.0','y2.0','y3.0']
zs = np.array([[0,1,2],[3,4,5]])
dc = DataCube(zs,[xs,ys])
dc['a']
>>>array([0, 1, 2])
dc['y1.0']
>>>array([0, 3])
dc['b','y1.0']
>>>3
dc['b','y1.0-y2.0']
>>>array([3, 4])
"""

import numpy as np
class DataCube:
    def __init__(self,data,string_lists):
        self.data = data
        if isinstance(data,np.ndarray):
            self.data = data
        elif isinstance(data,list):
            data = np.array(data)
            self.data = data
        else:
            assert False,'Array-shaped data is required'
        self.string_lists = string_lists
        assert len(string_lists) == len(data.shape),'len(string_lists) must equal len(data.shape),'+\
                    'yours were %d and %d'%(len(string_lists),len(data.shape))
        for i,l in enumerate(string_lists):
            assert len(l) == data.shape[i],'column %d has length %d, but %d labels were given,'\
                            %(i,data.shape[i],len(l))
            for string in l:
                assert '-' not in string,'Not allowed to use variable names containing "-". [violator: %s]'%string
        self.string_list_names = [l[0] for l in string_lists]
        self.string_list_numbers = {}
        for i,name in enumerate(self.string_list_names):
            self.string_list_numbers[name] = i
        self.shape = self.data.shape
    def __getitem__(self,*args):
        to_return = self.data
        used = []
        if len(args) == 1 and isinstance(args[0],tuple):
            args = args[0]
        for arg in args:
            not_used = list(set(self.string_list_names)-set(used))
            str_list_indices_to_use = sorted([self.string_list_numbers[k] for k in not_used])
            str_lists_to_use = [self.string_lists[k] for k in str_list_indices_to_use]
            if '-' in arg:
                element0,element1 = None,None
                low,high = arg.split('-')
                for i,l in enumerate(str_lists_to_use):
                    if low in l:
                        element0 = l.index(low)
                        element1 = l.index(high)+1
                        assert element1>element0, '%s is not a valid continuum arg'%arg
                        break
                if element0 is None:
                    raise IndexError('arg %s not in any list of remaining elements: %s'%(arg,str_lists_to_use))
                element_access_str = ':,'*i+'%d:%d,'%(element0,element1)+':,'*(len(str_lists_to_use)-i-1)
            else:
                element = None
                for i,l in enumerate(str_lists_to_use):
                    if arg in l:
                        element = l.index(arg)
                        used.append(l[0])
                        break
                if element is None:
                    raise IndexError('arg %s not in any list of remaining elements: %s'%(arg,str_lists_to_use))
                element_access_str = ':,'*i+'%d,'%element+':,'*(len(str_lists_to_use)-i-1)
            to_return = eval('to_return[%s]'%element_access_str[:-1])
        return to_return
    def __repr__(self):
        return "<%d-D DataCube with categories %s>"%(len(self.string_lists),str(self.string_lists))
