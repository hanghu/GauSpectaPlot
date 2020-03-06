"""
  a library of utility functions for analyzing Gaussian log file 
"""
import numpy as np
import pandas as pd
import re

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
    foundIdentifier = False
    foundMatrix = False
    emptyRowId  = ' '*8
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
            if(len(line) < 8):
                if(mat_raw is None):
                    mat_raw = pd.DataFrame(cur_cols_raw,index=curRows,columns=curCols)
                else:
                    mat_raw = mat_raw.join(pd.DataFrame(cur_cols_raw,
                                                        index=curRows,columns=curCols))  
                break

            elif(line[:8] == emptyRowId):
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
                    iterRow = int(line[:7])
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


