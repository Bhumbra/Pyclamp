import inspect

def ret_members(x):
    for name in inspect.getmembers(x):
        print(name)
    
    
