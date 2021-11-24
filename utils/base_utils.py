import colorama
colorama.init()
from termcolor import colored, cprint

###########################################################################
############################ PRINT UTILS ##################################
###########################################################################
def main_proc_print(main_proc_str):
    (lambda x: cprint(x, 'yellow'))("[ MAIN PROCESS ] "+str(main_proc_str))
    

def warning_print(warning_str):
    (lambda x: cprint(x, 'red'))("[   WARNINGS   ] "+str(warning_str))

def result_print(result_str):
    (lambda x: cprint(x, 'green'))("[    OUTPUT    ] "+str(result_str))


def important_print(result_str):
    (lambda x: cprint(x, 'white',  'on_green'))("[    NOTICE    ] "+str(result_str))
