#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os, sys
import get_ei


def main():

    date = '2017-12'
    path_to_rep = '../'
    species = 'solenopsis_invicta'

    #            str         str      
    name, ei = get_ei.get_ei(path_to_rep,species,date)

    print (name, ei)


if __name__ == '__main__':

    main()



