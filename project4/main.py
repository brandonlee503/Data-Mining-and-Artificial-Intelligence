import sys
import numpy as np
import csv
import math
import operator
from itertools import combinations
from functools import reduce

from part1 import part1
from part2 import part2
from part2_test import part2Test
from shared import getData

def main():
    fName = 'data-2.txt'
    try:
        fName=sys.argv[1]
    except Exception as e:
        print('No data file provided, defaulting to {0}'.format(fName))
    data = getData(fName)

    # Part1
    part1(data)

    # Part2
    part2(data)


if __name__ == '__main__':
    main()
