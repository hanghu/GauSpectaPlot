"""
This module analyzes components of the molecular orbitals using a Gaussian output
log file (done with pop=full, iop(3/33=1).)
"""

import numpy as np
import pandas as pd
import re
from gLog_tools import read_matrix

class GroundState:
    '''
    Reads in and stor info about ground state from Gaussian log file.
    This includes the overlap matrix and detailed MO coefficients.
    '''

    def __init__(self,filename, ghf=True, rhf=True, MOdtype='complex'):
        '''
        FUTURE: Uses the filename to set basic attributes about
        the wavefunction and allocates appropriate space.
        '''
        #Read attributes
        assert isinstance(filename, str)
        assert isinstance(ghf, bool)
        assert isinstance(rhf,bool)
        assert MOdtype in ['Complex','complex','C','c',
                           'Real','real','R','r']

        self.filename = filename
        self.ghf = ghf
        self.rhf = rhf and ghf
        self._readBasisInfo()

        if MOdtype in ['Complex','complex','C','c']:
            self.MOdtype = np.complex128
        elif MOdtype in ['Real','real','R','r']:
            self.MOdtype = np.float64
        else:
            raise TypeError('Please specifiy MO data type as Complex or Real')
        
        self.overlapMatrix = None
        self.MOcoeffs = None
        self.AOtypes = None
        self.atomList = None
        self.HOMO = None
        self.LUMO = None
        self.eigenvalues = np.empty(self.Ne)
        # UHF Case: 
        if(not self.rhf): 
            self.MOcoeffs_b = None
            self.HOMO_b = None
            self.LUMO_b = None
            self.eigenvalues_b = None        

        return

    def _readBasisInfo(self):
        """
        Read number of basis and number of electrons
        """
        with open(self.filename, 'r') as f:
            foundBasisInfo = False
            line = f.readline().split()
            while not foundBasisInfo and line != '':
                if 'primitive' in line and 'gaussians,' in line:
                    self.nBas = int(line[0])
                    line = f.readline().split()
                    self.Ne = [int(line[0]), int(line[3])]
                    #print(self.nBas)
                    #print(self.Ne)
                    foundBasisInfo = True
                line = f.readline().split()

            assert foundBasisInfo

            if self.ghf:
                self.nSpinBas = 2*self.nBas
            else:
                self.nSpinBas = self.nBas

            print('NUM_OF_SPACICAL_BASIS=%i'  % self.nBas + ', ' + 
                  'NUM_OF_ALPHA_ELECTRONS=%i' % self.Ne[0] + ', ' +
                  'NUM_OF_BETA_ELECTRONS=%i'  % self.Ne[1])
        return

    def parseOverlaps(self):
        """
        Read overlap matrix 
        """
        #parser = cclib.io.ccopen(self.filename)
        #data = parser.parse()
        
        self.overlapMatrix = read_matrix(self.filename, 
                                         identifier=' *** Overlap *** ',
                                         match_method='p',
                                         matrix_format='lt')

        self.overlapMatrix = self.overlapMatrix + np.tril(self.overlapMatrix,k=-1).T

        if self.ghf:
            # expand overlap matrix to two component if ghf
            identity = np.array([[1.0,0.0],[0.0,1.0]],dtype=np.complex128)
            self.overlapMatrix = np.kron(self.overlapMatrix, identity)

        return

    def parseMOs(self):
        """
        Get MO Coefficients and parse it into different atoms associated with AO types
        
        Error_shooting:
            1.  'local variable 'atom' referenced before assignment':
                 check MOdtype, ghf, rhf input 
        """
        self.atomList = []
        self.AOtypes   = []
        

        if self.MOdtype == np.complex128 and self.ghf:
            nAOLines = 4
        elif self.MOdtype == np.complex128 and not self.ghf: 
            nAOLines = 2
        else:
            nAOLines = 1

        moEigenvalues = np.zeros(self.nSpinBas, dtype=np.float64)
        coeffs_df = pd.DataFrame(index=range(nAOLines*self.nBas))
        
        if not self.rhf:
            moEigenvalues_b = np.zeros(self.nSpinBas, dtype=np.float64)
            coeffs_df_b = pd.DataFrame(index=range(nAOLines*self.nBas))
        
        f = open(self.filename,'r')
        iterMO = 0
        iterAO = -1
        foundPopInfo   = False
        foundAOType    = False
        foundHOMOLUMO  = False
        parseBetaMOs = False
 
        for line in f:
            if not foundPopInfo:
                if len(line) > 10 and line[-10:-2] == 'l601.exe': 
                    foundPopInfo = True
                continue

            if iterMO == self.nSpinBas: 
                if(not self.rhf and not parseBetaMOs): 
                    parseBetaMOs = True
                    iterMO = 0
                    iterAO = -1
                else:
                    break

            if not foundHOMOLUMO:
                if re.search('occ. eigenvalues', line):
                    OCCs_stack = line.split()
                elif re.search('virt. eigenvalues', line):
                    if self.HOMO is None:
                        self.HOMO = float(OCCs_stack[-1])
                        self.LUMO = float(line[27:37])
                    
                    if not self.rhf and self.HOMO_b is None:
                        self.HOMO_b = float(OCCs_stack[-1])
                        self.LUMO_b = float(line[27:37])
                        foundHOMOLUMO = True
                    else:
                        foundHOMOLUMO = True
            
            if line[5:16] == 'Eigenvalues':
                nMOCurIter = int((len(line) - 21) / 10)
                # get the MO eigenvalues
                eigs = line[21:]
                iMO = iterMO
                MONums = []
                
                for i in range(nMOCurIter):
                    iEig = eigs[10*i:10*(i+1)]
                    iEig = float(iEig) if iEig[0] != '*' else np.nan
                    if not parseBetaMOs:
                        moEigenvalues[iMO] = iEig
                    else:
                        moEigenvalues_b[iMO] = iEig

                    iMO +=1
                    MONums.append(iMO)
                #print(MONums)
                iterAO = 0
                rawCoeffs = []

            # get the raw MO coefficients
            elif(iterAO >= 0):
                rawStr = line[21:]
                rawCoeffs.append([rawStr[10*i:10*(i+1)] for i in range(nMOCurIter)])
                # get AO type
                iterAO += 1

                if not foundAOType and iterAO%nAOLines == 0:
                    AOInfo = line[:21].split()
                    if self.ghf:
                        if len(AOInfo) > 4: 
                            atom = "".join(AOInfo[1:3])
                            quanNum = "".join(AOInfo[3:-1]) 
                        else:
                            quanNum = "".join(AOInfo[1:-1])
                    else:
                        if len(AOInfo) > 3:
                            atom = "".join(AOInfo[1:3])
                            quanNum = "".join(AOInfo[3:])
                        else:
                            quanNum = "".join(AOInfo[1:])
                    self.AOtypes.append([atom, quanNum])

                if(iterAO == nAOLines*self.nBas):
                    if not parseBetaMOs:
                        coeffs_df = coeffs_df.join(pd.DataFrame(rawCoeffs,columns=MONums))
                    else:
                        coeffs_df_b = coeffs_df_b.join(pd.DataFrame(rawCoeffs,columns=MONums))
                        
                    iterAO = -1
                    iterMO += nMOCurIter
                    if (not foundAOType): foundAOType = True

        f.close()
        
        if not foundPopInfo: raise ValueError('No MO coefficient in the log file')

        # assign eigenvalues
        self.eigenvalues = moEigenvalues
        coeffs_df = coeffs_df.applymap(float)

        if self.MOdtype == np.complex128:
            # combine the real and imaginary parts
            nMO = coeffs_df.values.shape[1]
            coeffs_raw = np.reshape(coeffs_df.values,
                                    (self.nSpinBas,2*nMO))
            coeffs_real = coeffs_raw[:, :nMO]
            coeffs_imag = coeffs_raw[:, nMO:]
            self.MOcoeffs = coeffs_real + coeffs_imag*1.0j
        else:
            self.MOcoeffs = coeffs_df.values

        if not self.rhf:
            self.eigenvalues_b = moEigenvalues_b
            coeffs_df_b = coeffs_df_b.applymap(float)

            if self.MOdtype == np.complex128:
                # combine the real and imaginary parts
                nMO = coeffs_df.values.shape[1]
                coeffs_raw = np.reshape(coeffs_df_b.values,
                                        (self.nSpinBas,2*nMO))
                coeffs_real = coeffs_raw[:, :nMO]
                coeffs_imag = coeffs_raw[:, nMO:]
                self.MOcoeffs_b = coeffs_real + coeffs_imag*1.0j
            else:
                self.MOcoeffs_b = coeffs_df_b.values

        new_aotypes = []
        for ao in self.AOtypes:
            if re.search(r'[XYZ]{2}',ao[1]):
                ao[1] = ao[1][:-2] + 'D' + ao[1][-2:]
            new_aotypes.append(ao)    
        self.AOtypes = new_aotypes 
        
        if self.ghf:
            # expand AOtypes
            AOtypes_a = [(ao[0] + ' a',ao[1]) for ao in self.AOtypes]
            AOtypes_b = [(ao[0] + ' b',ao[1]) for ao in self.AOtypes]
            self.AOtypes = list(map(lambda x: tuple(x),
                np.reshape(np.array([AOtypes_a, AOtypes_b]).T,
                           (2,self.nSpinBas)).T))

        print('MO Coefficients and AOTypes have been Obtained')
        return

    def gen_MOs_report(self, MO_list=None,E_range=None,report_detail=0,latex_format=False,
                       selected_atoms_groups=None,spin='a',return_analysis=False):
        """
        generate MO components report:

        Args:
            MO_list: list or a range of MOs to be analyzed. default is all MOs.
            report_details: 4-6  and 10-12 ony works for GHF
                0 ... no details, just a simple sanity check
                      C.conjugate * S * C = Idenity Matrix

                1 ... break down components to each type of atoms
                2 ... same as 1, and projected to atomic oritals of different angular momentum
                3 ... same as 1, and projected to differenct basis
                
                4 ... break down components to each type of atoms regarding to different spins
                5 ... same as 4, and projected to atomic oritals of different angular momentum
                6 ... same as 4, and projected to differenct basis

                7 ... break down components to each atom 
                8 ... same as 7, and projected to atomic oritals of different angular momentum
                9 ... same as 7, and projected to differenct basis
                
               10 ... break down components to each atom regarding to different spins
               11 ... same as 10, and projected to atomic oritals of different angular momentum
               12 ... same as 10, and projected to differenct basis

            latex_format: boolean, if True a latex table code will be produced
            spin: for RHF and GHF, it should be 'Alpha' all the time,
                  for UHF, it could be 'Alpha' or 'Beta'
            return_analysis: retrun the report as an array of dict, no printing will be generated
        """
        if self.MOcoeffs is None: self.parseMOs()
        if self.overlapMatrix is None: self.parseOverlaps()
        if MO_list is None: MO_list = range(1,self.MOcoeffs.shape[1]+1)
        
        spin = spin if not self.ghf and not self.rhf else 'a'

        if spin in ['A','a','Alpha','alpha']:
            MOcoeffs = self.MOcoeffs
            eigenvalues = self.eigenvalues
        elif spin in ['B', 'b', 'Beta', 'beta']:
            print('Doing the Beta Part Analysis')
            MOcoeffs =self.MOcoeffs_b
            eigenvalues = self.eigenvalues_b
        else:
            raise TypeError('Wrong Spin Type')
        
        if E_range is not None:
            assert len(E_range) == 2 and E_range[0] <= E_range[1]
            new_MO_list = []
            for mo in MO_list:
                #print(mo)
                E_mo =  eigenvalues[mo-1] 
                if E_mo >= E_range[0] and  E_mo <= E_range[1]: new_MO_list.append(mo) 
            # assert it's energy range
            if len(new_MO_list) == 0: return
            MO_list = new_MO_list

        # if there is selected atosm
        SAG_On = True if selected_atoms_groups is not None else False
        latex_format = latex_format and not return_analysis
        if SAG_On:
            assert isinstance(selected_atoms_groups, dict)
            SAG_range_vec = {}
            for label, atoms in selected_atoms_groups.items():
                SAG_range_vec[label] =  np.array([int(re.search(r'[0-9]+', ao[0]).group(0)) 
                                                  in atoms for ao in self.AOtypes]).astype(float)
        
        # get atoms info
        if report_detail > 0:
            sumAllSameAtom  = True if report_detail < 7 else False
            detailedSpin    = True if ((report_detail > 3 and report_detail <7) or
                                       report_detail > 9) else False
            detailedSpin    = detailedSpin and self.ghf
            detailedAngular = True if (report_detail%3 == 2) else False
            detailedBasis   = True if (report_detail%3 == 0) else False

        if sumAllSameAtom:
            AOtypes = [ (re.search(r'[^0-9]+', ao[0]).group(0), ao[1]) 
                        for ao in self.AOtypes]
        else:
            AOtypes = self.AOtypes
        
        if detailedBasis:
            angtypes = list(map(
                lambda x: re.search(r'[0-9]+[SPDFGHIJ]', x[1]).group(0),AOtypes))
            unique_angtypes = sorted(set(angtypes), key=angtypes.index)
            angtypes = np.array(angtypes)
        else:
            # get orbital info
            if detailedAngular:
                angtypes = list(map(
                    lambda x: re.search(r'[SPDFGHIJ]', x[1]).group(0),AOtypes))
                unique_angtypes = sorted(set(angtypes), key=angtypes.index)
                angtypes = np.array(angtypes)
        
        if report_detail > 0:
            if(self.ghf and not detailedSpin):
                atoms = list(map(lambda x: x[0][:-2], AOtypes))
            else:
                atoms = list(map(lambda x: x[0], AOtypes))
        
            unique_atoms = sorted(set(atoms), key=atoms.index)
            atoms = np.array(atoms)

        if return_analysis:
            results = []
        elif latex_format:
            # print table header
            print_info = '    MO# & atom & total '
            for angtype_i in unique_angtypes:
                print_info += ' & ' + angtype_i
            print_info += ' \\'+'\\'
            print(print_info)
        
        for MO in MO_list:
            colvec = MOcoeffs[:,MO-1].copy()
            overlap_dot_colvec = np.dot(self.overlapMatrix,colvec)
            rowvec = np.conjugate(colvec.T)
            value = np.dot(rowvec,overlap_dot_colvec)
            
            if return_analysis:
                MO_result = {'MO':MO}
                MO_result['Total Projection'] = value.real
                MO_result['Energy'] = eigenvalues[MO-1]
            elif not latex_format:
                print('>> MO %i : %7.4f; Energy: %7.5f' % (MO,value.real,eigenvalues[MO-1]))
            
            if report_detail > 0:
                if(latex_format):
                    print_MO_number = True
                    num_space = 0

                for atom_i in unique_atoms:
                    atom_range_vec = (atoms == atom_i).astype(float)
                    rowvec = np.conjugate((colvec * atom_range_vec).T)
                    value = np.dot(rowvec,overlap_dot_colvec)
                    
                    if return_analysis:
                        MO_result[str(atom_i)] = value.real
                    else:
                        print_info = '    '

                        if(latex_format):
                            if(print_MO_number):
                                print('    \\' + 'hline')
                                print_info += '\\' + 'multirow{' + str(len(unique_atoms)) \
                                    + '}{*}{' + str(MO)+'}'
                                num_space = len(print_info) - 4
                                print_MO_number = False
                            else:
                                print('    \\' + 'cline' + '{2-' + str(len(unique_angtypes)+3) + '}')
                                print_info += ' '*num_space
                            print_info += ' & %s & %7.4f' %(atom_i, value.real) 
                        else:
                            print_info += '---- %4s : %7.4f' % (atom_i, value.real)

                    if detailedAngular:
                        if return_analysis:
                            angular_results = {}
                        else:
                            if not latex_format: print_info += ' >>>>'
                            prefix_len = len(print_info)
                            item_i = 0
                        
                        for angtype_i in unique_angtypes:
                            angtype_range_vec = (angtypes == angtype_i).astype(float)
                            rowvec_i = rowvec * angtype_range_vec
                            value_i = np.dot(rowvec_i,overlap_dot_colvec)
                            
                            if return_analysis:
                                angular_results[angtype_i] = value_i.real
                            else:   
                                if (item_i > 4 and item_i%5==0):
                                    print_info += '\n' + ' '*prefix_len
                                if(latex_format):
                                    print_info += ' & %7.4f' % (value_i.real) 
                                else:
                                    print_info += ' %4s : %7.4f;' % (angtype_i, value_i.real)
                                item_i += 1
                    
                        if return_analysis:
                            MO_result[str(atom_i)] = [MO_result[str(atom_i)],angular_results]
                        

                    if not return_analysis:
                        if latex_format: print_info += ' \\'+'\\'
                        print(print_info)
                
                if SAG_On:
                    for label, atom_range_vec in SAG_range_vec.items():
                        rowvec = np.conjugate((colvec * atom_range_vec).T)
                        value = np.dot(rowvec,overlap_dot_colvec)
                        
                        if return_analysis:
                            MO_result[label] = value.real
                        else:
                            print_info = '    '
                            if(latex_format):
                                print('    \\' + 'cline' + '{2-' + str(len(unique_angtypes)+3) + '}')
                                print_info += ' '*num_space
                                print_info += ' & %s & %7.4f' %(label, value.real) 
                            else:
                                print_info += '---- %4s : %7.4f' % (label, value.real)
                        
                        if detailedAngular:
                            if return_analysis:
                                angular_results = {}
                            else:
                                if not latex_format: print_info += ' >>>>'
                                item_i = 0
                            
                            for angtype_i in unique_angtypes:
                                angtype_range_vec = (angtypes == angtype_i).astype(float)
                                rowvec_i = rowvec * angtype_range_vec
                                value_i = np.dot(rowvec_i,overlap_dot_colvec)
                                
                                if return_analysis:
                                    angular_results[angtype_i] = value_i.real
                                else:   
                                    if (item_i > 4 and item_i%5==0):
                                        print_info += '\n' + ' '*prefix_len
                                    if(latex_format):
                                        print_info += ' & %7.4f' % (value_i.real) 
                                    else:
                                        print_info += ' %4s : %7.4f;' % (angtype_i, value_i.real)
                                    item_i += 1
                                
                            if return_analysis:
                                MO_result[label] = [MO_result[label],angular_results]

                        if not return_analysis:
                            if latex_format: print_info += ' \\'+'\\'
                            print(print_info)
            
            if return_analysis: results.append(MO_result)
        
        if return_analysis:
            return results
        else:
            return 
        
#---------------------------------------------
# Running as a Script
#---------------------------------------------

if __name__ == '__main__':
    filename = 'ga_x2c.log'
    # TODO: auto detection 
    # For GHF: ghf=True,rhf=True,MOdtype='complex'
    # For RHF: ghf=False,rhf=True,MOdtype='real' 
    # For UHF: ghf=False,rhf=False,MOdtype='real' 
    gs = GroundState(filename,ghf=True,rhf=True,MOdtype='complex')
    #gs.parseMOs()
    #print(gs.MOcoeffs.shape)
    gs.gen_MOs_report(report_detail=2)
