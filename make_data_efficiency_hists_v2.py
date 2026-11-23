#!/usr/bin/env python
import os
import glob
import time
import argparse
import copy
import re
import numpy as np
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
import json
import yaml
import ROOT

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

ROOT.PyConfig.IgnoreCommandLineOptions = True


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


def get_input_files(inputs, n_files=None):
    def _process(p):
        if '*' in str(p):
            return sorted(p.parent.glob(p.name))
        if p.is_dir():
            return sorted(p.glob('*.root'))
        if p.is_file() and p.suffix == '.root':
            return [p]
        return []

    files = sorted(set(str(f) for path in ([inputs] if isinstance(inputs, str) else inputs)
                       for f in _process(Path(path))))
    return files[:n_files] if n_files and n_files <= len(files) else files


class DataTriggerEfficiencyProducer(Module):
    def __init__(self, params):
        self.params = DotDict(params)
        self.writeHistFile = True
        self.trigger_map = self.params.trigger_map
        self.trigger_map['trigger_OR'] = 0


    def make_th1(self, name, xbins):
        h = ROOT.TH1F(name, name, len(xbins)-1, xbins)
        self.addObject(h)
        return h


    def make_th2(self, name, xbins, ybins):
        h = ROOT.TH2F(name, name, len(xbins)-1, xbins, len(ybins)-1, ybins)
        self.addObject(h)
        return h


    def fill_th1(self, h, arr, w):
        h.Fill(arr, np.ones_like(arr)*w)


    def fill_th2(self, h, arr_x, arr_y, w):
        shape = np.broadcast_shapes(np.shape(arr_x), np.shape(arr_y))
        h.Fill(arr_x, arr_y, (np.ones_like(shape) if shape else 1)*w)


    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)

        # Hist Binnings
        self.diel_m_bins = np.linspace(2, 4, 100, dtype=np.double)
        self.pt_bins     = np.array([5, 7, 9, 10, 11, 12, 13, 999], dtype=np.double)
        # self.pt_bins     = np.array([5, 8, 11, 999], dtype=np.double)
        self.dr_bins     = np.array([0, 0.12, 0.2, 0.28, 0.44, 1.], dtype=np.double)
        self.npv_bins    = np.linspace(0, 100, 100, dtype=np.double)

        # 1D Hists
        self.h_npv_dict    = {i : self.make_th1(f'npv_{i}', self.npv_bins) for i in self.trigger_map.keys()}

        # Pt Eff
        self.h_diel_m_num_ptbinned_dict      = {i : self.make_th2(f'diel_m_{i}_num_ptbinned', self.diel_m_bins,  self.pt_bins) for i in self.trigger_map.keys()}
        self.h_diel_m_denom_ptbinned_dict    = {i : self.make_th2(f'diel_m_{i}_denom_ptbinned', self.diel_m_bins,  self.pt_bins) for i in self.trigger_map.keys()}


    def analyze(self, event):
        # Define Physics Objects
        electrons = Collection(event, 'Electron')
        trig_L1   = Object(event, 'L1')
        trig_HLT  = Object(event, 'HLT')
        pv        = Object(event, 'PV')


        # Define Kinematic Variables
        # lead_el_pt    = electrons[0].pt
        sublead_el_pt = electrons[1].pt
        # sublead_eta   = electrons[1].eta
        # sublead_phi   = electrons[1].phi
        # dr            = electrons[0].DeltaR(electrons[1])
        # diel_pt       = (electrons[0].p4() + electrons[1].p4()).Pt()
        diel_m        = (electrons[0].p4() + electrons[1].p4()).M()

        # Define Trigger Paths
        trig_bit_dict = {k : (getattr(trig_L1, v[0]) and getattr(trig_HLT, v[1]) if v else 0) for k,v in self.trigger_map.items()}
        trig_bit_dict['trigger_OR'] = any(i for i in trig_bit_dict.values())

        # Trigger Efficiencies
        for path in trig_bit_dict.keys():
            self.fill_th2(
                self.h_diel_m_denom_ptbinned_dict[path],
                diel_m, 
                sublead_el_pt,
                1
            )

            if trig_bit_dict[path]:
                self.fill_th2(
                    self.h_diel_m_num_ptbinned_dict[path],
                    diel_m, 
                    sublead_el_pt,
                    1
                )

                self.fill_th1(
                    self.h_npv_dict[path],
                    pv.npvs,
                    1
                )
        return True


def worker(params):
    p = PostProcessor(
            params['output_dir'],
            params['input_files'],
            cut=params['presel'],
            branchsel=None,
            modules=[DataTriggerEfficiencyProducer(params)],
            noOut=True,
            histDirName='hists',
            histFileName=str(params['output_file']),
    )
    p.run()


def main(cfg):
    global_cfg = DotDict(cfg.global_cfg)

    out_path = Path(global_cfg.output_dir)
    if global_cfg.test:
        out_path = out_path.parent / 'test'
    os.makedirs(out_path, exist_ok=True)

    job_configs = []
    for job in cfg.data_samples:
        job = DotDict(job)
        if (global_cfg.test and ('test' not in job.name)) or (not global_cfg.test and ('test' in job.name)):
            continue
        job.input_files = get_input_files(job.inputs)
        job.output_dir = out_path
        job.output_file = out_path / f'effs_{job.name}.root'
        job.trigger_map = cfg.trigger_map
        job_configs.append(copy.deepcopy(job))

    if 'mp' in global_cfg.run_strategy:
        start_time = time.perf_counter()

        n_cores = mp.cpu_count() // 2
        with mp.Pool(processes=n_cores) as pool:
            pool.map(worker, [job.to_dict() for job in job_configs])

        finish_time = time.perf_counter()
        print(f'Finished in {finish_time - start_time} seconds')

    else:
        start_time = time.perf_counter()
        for job in job_configs:
            worker(job)

        finish_time = time.perf_counter()
        print(f'Finished in {finish_time - start_time} seconds')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_hist_cfg_v2.yml', help='configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))
    cfg.global_cfg.test = args.test

    main(cfg)
