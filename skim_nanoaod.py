#!/usr/bin/env python

import multiprocessing as mp
import time
import json
import os
import glob
import sys
import shutil
from importlib import import_module
from itertools import combinations
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from input_files import *
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

class SkimEvents(Module):
    def __init__(self):
        self.writeHistFile = True
    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)
    def analyze(self, event):
        return True

def worker(params, nfiles=None):
    idx = params['idx'] if 'idx' in params else None
    pf = ''.join(['_',str(idx),'_skim']) if idx else '_skim'
    infiles = glob.glob(params['input_files']) if nfiles is None else glob.glob(params['input_files'])[:nfiles]
    p = PostProcessor(
            params['output_dir'], 
            infiles,
            cut=params['presel'],
            branchsel=None, 
            modules=[SkimEvents()],
            jsonInput=params['json'],
            prefetch=True,
            longTermCache=True,
            postfix=pf
    )
    p.run()

if __name__=='__main__':

    # Prelection Cuts
    preselection_data = 'nElectron > 1 \
        && Electron_pt[0] > 5. \
        && Electron_pt[1] > 5. \
        && abs(Electron_eta[0]) < 1.22 \
        && abs(Electron_eta[1]) < 1.22 \
        && Electron_charge[0] + Electron_charge[1] == 0 \
        && (HLT_DoubleMu4_3_Bs \
            || HLT_DoubleMu4_3_Jpsi \
            || HLT_DoubleMu4_3_LowMass \
            || HLT_DoubleMu4_LowMass_Displaced \
            || HLT_Mu0_L1DoubleMu \
            || HLT_Mu4_L1DoubleMu \
            || HLT_DoubleMu3_Trk_Tau3mu \
            || HLT_DoubleMu3_TkMu_DsTau3Mu \
            || HLT_DoubleMu4_MuMuTrk_Displaced \
            || HLT_DoubleMu4_Jpsi_Displaced \
            || HLT_DoubleMu4_Jpsi_NoVertexing \
            || HLT_DoubleMu4_JpsiTrkTrk_Displaced \
            || HLT_DoubleMu4_JpsiTrk_Bc \
            || HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass \
            || HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05)'

    preselection_mc = 'nElectron > 1 \
        && Electron_pt[0] > 5. \
        && Electron_pt[1] > 5. \
        && abs(Electron_eta[0]) < 1.22 \
        && abs(Electron_eta[1]) < 1.22 \
        && Electron_charge[0] + Electron_charge[1] == 0'

    input_config = [
        # {'name' : 'test',
        #  'input_files' : ['/eos/uscms/store/user/nzipper/ParkingDoubleMuonLowMass7/NanoTestPost/220905_151031/0001/output_1179.root'],
        #  'output_dir'  : '.',
        #  'json_path'   : '/uscms/home/nzipper/nobackup/Rk/Analysis/CMSSW_10_4_0/src/PhysicsTools/NanoAODTools/JSON/Eras_C_Dv1_Dv2',
        #  'presel'      : 'true',
        # },
        # {'name'        : 'BuToKee',
        #  'input_files' : ,
        #  'output_dir'  : ,
        #  'json_path'   : None,
        #  'presel'      : preselection_mc,
        # },
        # {'name'        : 'BuToKJpsi_Toee',
        #  'input_files' : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/BuTOjpsiKEE/*.root',
        #  'output_dir'  : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/BuTOjpsiKEESkims',
        #  'json_path'   : None,
        #  'presel'      : preselection_mc,
        # },
        # {'name'        : '2022C',
        #  'input_files' : input_files_2022C,
        #  'output_dir'  : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022C',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : preselection_data,
        # },
        # {'name'        : '2022D',
        #  'input_files' : input_files_2022D,
        #  'output_dir'  : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022D',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : preselection_data,
        # },
        {'name'        : '2022E',
         'input_files' : input_files_2022E,
         'output_dir'  : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022E',
         'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
         'presel'      : preselection_data,
        },
        # {'name'        : '2022F',
        #  'input_files' : input_files_2022F,
        #  'output_dir'  : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022F',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : preselection_data,
        # },
        # {'name'        : '2022G',
        #  'input_files' : input_files_2022G,
        #  'output_dir'  : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022G',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : preselection_data,
        # },
    ]

    # Parameters
    runStrategy = 'mp'

    job_configs = []
    for params in input_config:
        if params['json_path'] is None:
            job_dict = {}
            job_dict['json'] = None 
            job_dict['output_dir'] = params['output_dir']
            job_dict['input_files'] = params['input_files'] 
            job_dict['presel'] = params['presel']
            job_configs.append(job_dict)
        else:
            if isinstance(params['input_files'], list):
                for idx, input_files in enumerate(params['input_files']):
                    job_dict = {}
                    job_dict['json'] = None
                    job_dict['output_dir'] = ''.join([params['output_dir'] , '/Preselection'])
                    job_dict['input_files'] = input_files 
                    job_dict['presel'] = params['presel']
                    job_dict['idx'] = idx
                    job_configs.append(job_dict)

            else:
                job_dict = {}
                job_dict['json'] = None
                job_dict['output_dir'] = ''.join([params['output_dir'] , '/Preselection'])
                job_dict['input_files'] = input_files 
                job_dict['presel'] = params['presel']
                job_dict['idx'] = idx
                job_configs.append(job_dict)

    if 'mp' in runStrategy:
        n_cores = mp.cpu_count()
        print(''.join(['Distributing ~',str(len(job_configs)),' jobs to ', str(n_cores), ' cores...']))

        start_time = time.perf_counter()
        jobs = []

        for params in job_configs:
            pool = mp.Pool()
            proc = mp.Process(target=worker, args=(params, ))
            jobs.append(proc)
            proc.start()

        for p in jobs:
            p.join()

        finish_time = time.perf_counter()
        print(''.join(['Finished in ', str(finish_time-start_time), ' seconds']))

#     elif 'batch' in runStrategy:
#         if not os.path.exists('./tmp_condor'): 
#             os.makedirs('./tmp_condor')
#         else:
#             shutil.rmtree('./tmp_condor')
#             os.makedirs('./tmp_condor')

#         for params in job_configs:
#             with open('tmp_condor/condor_job_cfg.json','w+') as f:
#                 f.write(json.dumps(params))
#             with open('tmp_condor/condor_job_exe.sh','w+') as f:
#                 f.write('#!/bin/bash \n')
#                 f.write('mkdir -p PhysicsTools/NanoAODTools/postprocessing/framework \n')
#                 f.write('cp python/postprocessing/framework/* PhysicsTools/NanoAODTools/postprocessing/framework \n')
#                 f.write('touch PhysicsTools/NanoAODTools/postprocessing/framework/__init__.py \n')
# #                 f.write(
# #                     """
# # python -c '
# # import imp
# # eventloop = imp.load_source("eventloop", "python/postprocessing/framework/eventloop.py")'
# #                     """
# #                     )
# #                 f.write(
# #                     """
# # python -c '
# # import sys
# # import os
# # python -c 'import json
# # import SkimEvents
# # f = open("condor_job_cfg.json")
# # params = json.load(f)
# # SkimEvents.worker(params)
# # f.close()'
# #                     """
# #                     )
#             os.chmod('tmp_condor/condor_job_exe.sh', 0o755)

#             print('Submitting {} - {} - {}'.format(params['output_dir'].split('/')[-2], params['output_dir'].split('/')[-1],params['idx']))
#             os.system('condor_submit tmp_condor/SkimEvents.sub')

    else:
        for params in job_configs:
            worker(params)
