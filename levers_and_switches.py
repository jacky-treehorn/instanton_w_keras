import itertools
import numpy as np
import re

class inp_reader:
  def __init__(self, filename):
    self.filename = filename
  def readlines(self,split_at): #split_at is a list of symbols, will split lines in the file recursively.
    fhand = open(self.filename)
    line_set = []
    count = 0
    for line in fhand:
      line_set.append(line.rstrip())
      count +=1
    self.line_set = line_set
    line_set_split = line_set
    if(len(split_at)>0):
      for char in split_at:
        tmp = []
        for elem in line_set_split:
          tmp.append(elem.split(char))
        line_set_split = list(itertools.chain.from_iterable(tmp))
      tmp = []
      for elem in line_set_split:
        tmp.append(elem.rstrip().lstrip())
      line_set_split = np.asarray(tmp)
      self.line_set_split=line_set_split

re_f_float_neg = re.compile('(-?[0-9.]*)(-\d\d\d)')
def fortran_float(input_string):
    """
    Return a float of the input string, just like `float(input_string)`,
    but allowing for Fortran's string formatting to screw it up when 
    you have very small numbers (like 0.31674-103 instead of 0.31674E-103 )
    """
    try:
        fl = float(input_string)
    except ValueError,e:
        match = re_f_float_neg.match(input_string.strip())
        if match:
            processed_string = match.group(1)+'E'+match.group(2)
            fl = float(processed_string)
        else:
            raise e
    return fl

def inp_converter(configuration):
  configuration = configuration[::2]
  tmp = []
  for elem in configuration:
    tmp.append(elem.split(','))
  configuration = tmp
  tmp = []
  for elem in configuration:
    tmp2 = []
    for item in elem:
      try:
        tmp2.append(fortran_float(item.replace('d','E')))
      except:
        tmp2.append(item)
    tmp.append(tmp2)
  return tmp
