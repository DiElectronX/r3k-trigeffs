import os
import sys
import multiprocessing as mp
import argparse
import random
import string
import time
import json
import yaml
import glob
import ROOT
from pathlib import Path
import copy
from pprint import pprint
from datetime import datetime
import shutil

sys.path.insert(0, os.path.abspath('crab_skimmer'))
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

from CRABAPI.RawCommand import crabCommand
from PhysicsTools.NanoAODTools.postprocessing.utils.crabhelper import inputFiles, runsAndLumis
from crab_skimmer.crab_cfg_template import config


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, key):
        return self.get(key, False)

    def __setattr__(self, key, value):
        self[key] = DotDict(value) if isinstance(value, dict) else value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f'No attribute named {key}')

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo))

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, DotDict) else v for k, v in self.items()}


class SkimEvents(Module):
    def __init__(self):
        self.writeHistFile = True
    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)
    def analyze(self, event):
        return True


def worker(params):
    pf = f'_skim_{"".join(random.choices(string.ascii_lowercase + string.digits, k=8))}'
    p = PostProcessor(
            params['output_path'],
            params['files'],
            cut=params['preselection'],
            branchsel=None,
            modules=[SkimEvents()],
            jsonInput=params['json'],
            prefetch=True,
            longTermCache=True,
            postfix=pf
    )
    p.run()


def main(cfg):
    datasets = {k:v for k,v in cfg.datasets.items() if k in cfg.sel_datasets} if cfg.sel_datasets else cfg.datasets
    assert datasets, 'No valid dataset given!'
    job_configs = []
    for job_name, params in datasets.items():
        if cfg.test and ('test' not in job_name):
            continue
        elif not cfg.test and ('test' in job_name):
            continue
        job_dict = DotDict({
            'name' : job_name,
            'json' : params.json_path if params.json_path else None,
            'output_path' : params.output_path,
            'files' : [f for path in params.files for f in glob.glob(path, recursive=True)],
            'preselection' : params.preselection,
        })
        job_configs.append(copy.deepcopy(job_dict))

    start_time = time.perf_counter()
    if 'mp' in cfg['config']['run_strategy']:
        n_cores = mp.cpu_count()
        if cfg.verbose:
            print(''.join(['Distributing ~',str(len(job_configs)),' jobs to ', str(n_cores), ' cores...']))

        with mp.Pool(processes=n_cores) as pool:
            pool.map(worker, [job.to_dict() for job in job_configs])

    if 'batch' in cfg['config']['run_strategy']:
        for params in job_configs:
            config.General.workArea = 'crab_skimmer/crab_jobs'
            config.General.requestName = '_'.join([params.name,datetime.now().strftime("%m_%d_%y")])
            config.Data.userInputFiles = [f.replace('/eos/cms/', 'root://cmsxrootd.fnal.gov//') for f in params.files]
            config.Data.unitsPerJob = 200
            config.Data.totalUnits = len(params.files)
            config.Data.outLFNDirBase = params.output_path[params.output_path.index('/store'):]

            config_values = {
                'CUT_TEMPLATE'  : f"'{params.preselection}'" if params.preselection is not None else None,
                'JSON_TEMPLATE' : f"'{os.sep.join(params.json.split(os.sep)[1:])}'" if params.json is not None else None,
            }

            with open('crab_skimmer/crab_script_template.py', 'r') as f:
                content = f.read().format(**config_values)

            with open('crab_skimmer/crab_script.py', 'w') as f:
                f.write(content) 

            request_path = Path(config.General.workArea) / ('crab_'+config.General.requestName)
            if request_path.exists():
                if input(f"CRAB request '{request_path}' exists. Overwrite? (y/n): ").strip().lower() == 'y':
                    shutil.rmtree(request_path)
                else:
                    raise FileExistsError('CRAB request already exists. Try a job with another name.')

            res = crabCommand('submit', config=config)
            print(f'Submitted job {params.name} to CRAB batch system\n',res)
    else:
        for params in job_configs:
            worker(params)

    if cfg.verbose:
        finish_time = time.perf_counter()
        print(''.join(['Finished in ', str(round(finish_time-start_time)), ' seconds']))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_skim_cfg.yml', help='skim configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-d', '--datasets', dest='datasets', nargs='+', help='target datasets to run (from cfg file)')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))
    
    cfg.verbose = args.verbose
    cfg.sel_datasets = args.datasets
    cfg.test = args.test
    
    main(cfg)
