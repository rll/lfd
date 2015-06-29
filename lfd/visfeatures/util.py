#!/usr/bin/env python

def tuplify(x):                                                                 
    if isIterable(x):                                                           
        return tuple([tuplify(y) for y in x])                                   
    else:                                                                       
        return x

def isIterable(x):                                                              
    if type(x) in (str, unicode):                                               
        return False                                                            
    try:                                                                        
        x_iter = iter(x)                                                        
        return True                                                             
    except:                                                                     
        return False 
