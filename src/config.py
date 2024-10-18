# define global variable to pass into scipy optimizer
global OPT_INFO

OPT_INFO = {                     
    'fold': None,                   # shared
    'opt_iter': None,             
    'shape_V': None,                # batch
    'opt_history': None,        
    'col_num':  None,               # iterative 
    'Q': None,
    'last_opt_history': None,
    'opt_histories': None
}