import multiprocessing as mp
import argparse
import time
import json
import yaml
import glob
import ROOT

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor


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
    infiles = glob.glob(params['files']) if nfiles is None else glob.glob(params['files'])[:nfiles]
    p = PostProcessor(
            params['output_path'],
            infiles,
            cut=params['preselection'],
            branchsel=None,
            modules=[SkimEvents()],
            jsonInput=params['json'],
            prefetch=True,
            longTermCache=True,
            postfix=pf
    )
    p.run()


def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    datasets = {k:v for k,v in cfg['datasets'].items() if k in args.datasets} if args.datasets else cfg['datasets']

    job_configs = []
    for params in datasets.values():
        if params['json_path'] is None:
            job_dict = {}
            job_dict['json'] = None
            job_dict['output_path'] = params['output_path']
            job_dict['files'] = params['files']
            job_dict['preselection'] = params['preselection']
            job_configs.append(job_dict)
        else:
            if isinstance(params['files'], list):
                for idx, files in enumerate(params['files']):
                    job_dict = {}
                    job_dict['json'] = None
                    job_dict['output_path'] = ''.join([params['output_path'] , '/Preselection'])
                    job_dict['files'] = files
                    job_dict['preselection'] = params['preselection']
                    job_dict['idx'] = idx
                    job_configs.append(job_dict)
            else:
                job_dict = {}
                job_dict['json'] = None
                job_dict['output_path'] = ''.join([params['output_path'] , '/Preselection'])
                job_dict['files'] = params['files']
                job_dict['preselection'] = params['preselection']
                job_dict['idx'] = 0
                job_configs.append(job_dict)

    if 'mp' in cfg['config']['run_strategy']:
        n_cores = mp.cpu_count()
        if args.verbose:
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
        if args.verbose:
            print(''.join(['Finished in ', str(finish_time-start_time), ' seconds']))
    else:
        for params in job_configs:
            worker(params)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='skim_cfg.yml', help='skim configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-d', '--datasets', dest='datasets', nargs='+', help='target datasets to run (from cfg file)')
    args = parser.parse_args()

    main(args)
