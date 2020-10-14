"""
  a library of utility functions for analyzing Gaussian log file 
"""
import numpy as np
import pandas as pd
import re

def match_full_line(pattern,line):
    return line == pattern

def match_partial_line(pattern, line):
    return len(line) >= len(pattern) and line[0:len(pattern)] == pattern

def search_line_regex(pattern, line):
    return re.serach(pattern,line)

def read_log_block(filename,start_id,end_id=None,match_method='full line',
                   N_block=None,n_lines_per_block=None):
    """simplied 'grep' in python"""

    if match_method in ['F', 'f', 'full line']:
        match_func = match_full_line
    elif match_method in ['P', 'p', 'partial line']:
        match_func = match_partial_line
    elif match_method in ['R', 'r', 'regex', 'regular expression']:
        match_func = search_line_regex
    elif callable(match_method):
        match_func = match_method
    else:
        raise TypeError('Match method are not defined')

    if end_id is None and n_lines_per_block is None:
        n_lines_per_block = 1
    
    f = open(filename,'r')
    foundBlock = False
    blocks = []
    line_count = 0 if N_block is not None else None
    
    for line in f:
        if len(blocks) == N_block: break

        if not foundBlock:
            if match_func(start_id, line):
                foundBlock=True
                block_i = [line]
            continue

        if foundBlock:
            if end_id is not None and match_func(end_id, line):
                block_i.append(line)
                blocks.append(block_i)
                foundBlock = False
                continue
            
            if len(block_i) == n_lines_per_block:
                blocks.append(block_i)
                foundBlock = False
                continue

            block_i.append(line)

    f.close()
    return blocks

def gaustr_to_num(x,convert_func=float):
    if not isinstance(x, str): return x
    
    if re.search(r'D', x):
        x = x.replace('D','E')
    elif re.fullmatch(r'[0-9]+\-[0-9]+', x):
        x = x.replace('-','E-')
    
    return convert_func(x)

def read_matrix(filename, identifier, matrix_format='full'):
    """
        Module to read a printed matrix from gaussian output log 
        file with a identifier line. Note that identifier must 
        be a complete line before a printed Matrix.
    """

    assert isinstance(filename, str)
    assert isinstance(identifier, str)
    assert matrix_format in ['full','F', 'f', 
                             'upper triangle','UT','ut', 
                             'lower triangle','LT', 'lt'] 
    
    f = open(filename,'r')
    lenRowId = 7
    foundIdentifier = False
    foundMatrix = False
    emptyRowId  = ' '*lenRowId
    mat_raw = None
    identifier += '\n'

    for line in f:

        if not foundIdentifier:
            if (line == identifier): foundIdentifier = True
            continue
        
        if(not foundMatrix and foundIdentifier):
            exam_line = line.split()
            if len(exam_line)== 0: continue
            start_indicator = [str(i+1) for i in range(len(exam_line))]
            if exam_line != start_indicator: continue
            
            foundMatrix = True
            first = True
            firstFinished = False
            nRow = 0

        if(foundMatrix):
            if(len(line) < lenRowId):
                if(mat_raw is None):
                    mat_raw = pd.DataFrame(cur_cols_raw,index=curRows,columns=curCols)
                else:
                    mat_raw = mat_raw.join(pd.DataFrame(cur_cols_raw,
                                                        index=curRows,columns=curCols))  
                break

            elif(line[:lenRowId] == emptyRowId):
               # Col number line
               if(first and firstFinished): first = False

               if(first):
                   firstFinished = True
               else:
                   #print(curCols)
                   if(mat_raw is None):
                       mat_raw = pd.DataFrame(cur_cols_raw,index=curRows,columns=curCols)
                   else:    
                       mat_raw = mat_raw.join(pd.DataFrame(cur_cols_raw,
                                                           index=curRows,columns=curCols))  

               cur_cols_raw = []
               curRows = []
               curCols = list(map(int,line.split()))
               if(len(curCols) == 0): 
                   last =True
                   break

            else: 
               # Row number line and value
                iterRow = -1
                
                try: # TODO: add function to deal with spin 
                    iterRow = int(line[:lenRowId])
                except ValueError:
                    #print(curCols)
                    if(mat_raw is None):
                        mat_raw = pd.DataFrame(cur_cols_raw,index=curRows,columns=curCols)
                    else:
                        mat_raw = mat_raw.join(pd.DataFrame(cur_cols_raw,
                                                            index=curRows,columns=curCols))  
                    break

                # Exam invalied conditions and add values
                if(first):
                    nRow += 1
                    if(iterRow != nRow): raise ValueError('Matrix found is incomplete')
                else:
                    if(iterRow > nRow): raise ValueError('Matrix row out of bound')

                curRows.append(iterRow) 
                cur_cols_raw.append(line.split()[1:])


    f.close()
    if(mat_raw is not None):
        print(identifier[:-1]+' has been obtained')
    else:
        print(identifier[:-1]+' are not found in the log file')
        return None
    
    if matrix_format in ['full','F', 'f']:
        mat_raw = mat_raw.values
    elif matrix_format in ['upper triangle','UT','ut']:
        mat_raw = np.triu(mat_raw.values)
    elif matrix_format in ['lower triangle','LT', 'lt']:
        mat_raw = np.tril(mat_raw.values) 
    
    return np.vectorize(gaustr_to_num)(mat_raw)


