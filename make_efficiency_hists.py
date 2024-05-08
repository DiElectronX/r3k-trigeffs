#!/usr/bin/env python
import os
import glob
import sys
import time
import numpy as np
from importlib import import_module
import multiprocessing
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

def assign_frac_trig(idx, lints, entries):
    frac_evts = np.around((lints / np.sum(lints) * entries))
    idxs = np.cumsum(frac_evts)
    return np.digitize(idx,idxs)

def isB(part):
    return (np.abs(part.pdgId) % 1000) // 100 == 5

def isJPsi(part):
    return np.abs(part.pdgId) == 443

def getBtoJPsi(parts, idx=False):
    j_idx = np.array(map(isJPsi, parts)).argmax()
    b_idx = parts[j_idx].genPartIdxMother
    if b_idx < 0: 
        out = None
    else: 
        out = b_idx if idx else parts[b_idx]
    return out

def getOtherB(parts, idx=False):
    excluded_Bs = []
    parentB = True
    b_idx = getBtoJPsi(parts, idx=True)
    if b_idx is not None:
        while parentB:
            excluded_Bs.append(b_idx)
            mother_idx = parts[b_idx].genPartIdxMother
            if mother_idx >= 0:
                if isB(parts[mother_idx]):
                    b_idx = mother_idx
                else: 
                    parentB = False
            else: parentB = False

        b_idxs = np.array(map(isB, parts))
        for i in excluded_Bs:
            b_idxs[i] = False

        otherB_idx = next((i for i, j in enumerate(b_idxs) if j), None)
        if otherB_idx is None: 
            out = None
        else:
            out = otherB_idx if idx else parts[otherB_idx]
    else: out = None
    return out


class TriggerEfficiencyProducer(Module):
    def __init__(self, trigger_paths, isMC=False):
        self.trigger_paths = trigger_paths
        self.isMC = isMC
        self.writeHistFile = True

    def make_th1(self, name, xbins):
        h = ROOT.TH1F(name, name, len(xbins)-1, xbins)
        self.addObject(h)
        return h

    def make_th2(self, name, xbins, ybins):
        h = ROOT.TH2F(name, name, len(xbins)-1, xbins, len(ybins)-1, ybins)
        self.addObject(h)
        return h

    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)

        # Hist Binnings
        self.diel_m_bins = np.linspace(2, 4, 100, dtype=np.double)
        self.pt_bins     = np.array([5, 7, 9, 10, 11, 12, 13, 999], dtype=np.double)
        self.eta_bins    = np.array([-1.22, -0.7, -.2, 0.2, .7, 1.22], dtype=np.double)
        self.dr_bins     = np.array([0, 0.12, 0.2, 0.28, 0.44, 1.], dtype=np.double)
        self.npv_bins    = np.array([0, 22, 27,31, 36, 100], dtype=np.double)
        self.diept_bins  = np.array([5, 10, 11, 12, 15, 20, 40, 70, 999], dtype=np.double)
        
        # Kinematic plots
        self.h_el_pt                        = ROOT.TH1F('el_pt', 'el_pt', 500, 0, 100); self.addObject(self.h_el_pt)
        self.h_el_eta                       = ROOT.TH1F('el_eta', 'el_eta', 100, -2, 2); self.addObject(self.h_el_eta)
        self.h_el_phi                       = ROOT.TH1F('el_phi', 'el_phi', 100, -4, 4); self.addObject(self.h_el_phi)
        self.h_lead_el_pt                   = ROOT.TH1F('lead_el_pt', 'lead_el_pt', 500, 0, 100); self.addObject(self.h_lead_el_pt)
        self.h_sublead_el_pt                = ROOT.TH1F('sublead_el_pt', 'sublead_el_pt', 500, 0, 100); self.addObject(self.h_sublead_el_pt)
        self.h_sublead_el_eta               = ROOT.TH1F('sublead_el_eta', 'sublead_el_eta', 100, -2, 2); self.addObject(self.h_sublead_el_eta)
        self.h_sublead_el_phi               = ROOT.TH1F('sublead_el_phi', 'sublead_el_phi', 100, -4, 4); self.addObject(self.h_sublead_el_phi)
        self.h_diel_m                       = ROOT.TH1F('diel_m', 'diel_m', 100, 2, 4); self.addObject(self.h_diel_m)
        self.h_dr                           = ROOT.TH1F('dr', 'dr', 100, 0, 4); self.addObject(self.h_dr)
        self.h_npv                          = ROOT.TH1F('npv', 'npv', 80, 0, 80); self.addObject(self.h_npv)
        self.h_diel_pt                      = ROOT.TH1F('diel_pt', 'diel_pt', 500, 0, 100); self.addObject(self.h_diel_pt)
        self.h2_sublead_el_pt_dr            = self.make_th2('sublead_el_pt_dr', self.pt_bins,  self.dr_bins)

        self.h_el_pt_trigger_or             = ROOT.TH1F('el_pt_trigger_or', 'el_pt_trigger_or', 500, 0, 100); self.addObject(self.h_el_pt_trigger_or)
        self.h_el_eta_trigger_or            = ROOT.TH1F('el_eta_trigger_or', 'el_eta_trigger_or', 100, -2, 2); self.addObject(self.h_el_eta_trigger_or)
        self.h_el_phi_trigger_or            = ROOT.TH1F('el_phi_trigger_or', 'el_phi_trigger_or', 100, -4, 4); self.addObject(self.h_el_phi_trigger_or)
        self.h_sublead_el_pt_trigger_or     = ROOT.TH1F('sublead_el_pt_trigger_or', 'sublead_el_pt_trigger_or', 500, 0, 100); self.addObject(self.h_sublead_el_pt_trigger_or)
        self.h_sublead_el_eta_trigger_or    = ROOT.TH1F('sublead_el_eta_trigger_or', 'sublead_el_eta_trigger_or', 100, -2, 2); self.addObject(self.h_sublead_el_eta_trigger_or)
        self.h_sublead_el_phi_trigger_or    = ROOT.TH1F('sublead_el_phi_trigger_or', 'sublead_el_phi_trigger_or', 100, -4, 4); self.addObject(self.h_sublead_el_phi_trigger_or)
        self.h_diel_m_trigger_or            = ROOT.TH1F('diel_m_trigger_or', 'diel_m_trigger_or', 100, 2, 4); self.addObject(self.h_diel_m_trigger_or)
        self.h_dr_trigger_or                = ROOT.TH1F('dr_trigger_or', 'dr_trigger_or', 100, 0, 4); self.addObject(self.h_dr_trigger_or)
        self.h_npv_trigger_or               = ROOT.TH1F('npv_trigger_or', 'npv_trigger_or', 80, 0, 80); self.addObject(self.h_npv_trigger_or)
        self.h2_sublead_el_pt_dr_trigger_or = self.make_th2('sublead_el_pt_dr_trigger_or', self.pt_bins,  self.dr_bins)

        # Trigger path PU 
        self.h_npvs =   [ self.make_th1('npv_'+str(trig), self.npv_bins) for trig in self.trigger_paths ] 

        # Pt Eff
        self.h_diel_m_num_ptbinned =   [ self.make_th2('diel_m_'+str(trig)+'_num_ptbinned', self.diel_m_bins,  self.pt_bins) for trig in self.trigger_paths ] 
        self.h_diel_m_denom_ptbinned = [ self.make_th2('diel_m_'+str(trig)+'_denom_ptbinned', self.diel_m_bins,  self.pt_bins) for trig in self.trigger_paths ]

        # Eta Eff 
        self.h_diel_m_num_etabinned =   [ self.make_th2('diel_m_'+str(trig)+'_num_etabinned', self.diel_m_bins,  self.eta_bins) for trig in self.trigger_paths ]
        self.h_diel_m_denom_etabinned = [ self.make_th2('diel_m_'+str(trig)+'_denom_etabinned', self.diel_m_bins,  self.eta_bins) for trig in self.trigger_paths ]

        # DR Eff 
        self.h_diel_m_num_drbinned =   [ self.make_th2('diel_m_'+str(trig)+'_num_drbinned', self.diel_m_bins,  self.dr_bins) for trig in self.trigger_paths ]
        self.h_diel_m_denom_drbinned = [ self.make_th2('diel_m_'+str(trig)+'_denom_drbinned', self.diel_m_bins,  self.dr_bins) for trig in self.trigger_paths ]

        # NPV Eff
        self.h_diel_m_num_npvbinned =   [ self.make_th2('diel_m_'+str(trig)+'_num_npvbinned', self.diel_m_bins,  self.npv_bins) for trig in self.trigger_paths ]
        self.h_diel_m_denom_npvbinned = [ self.make_th2('diel_m_'+str(trig)+'_denom_npvbinned', self.diel_m_bins,  self.npv_bins) for trig in self.trigger_paths ]

        # Di-E Pt Eff
        self.h_diel_m_num_dieptbinned =   [ self.make_th2('diel_m_'+str(trig)+'_num_dieptbinned', self.diel_m_bins,  self.diept_bins) for trig in self.trigger_paths ]
        self.h_diel_m_denom_dieptbinned = [ self.make_th2('diel_m_'+str(trig)+'_denom_dieptbinned', self.diel_m_bins,  self.diept_bins) for trig in self.trigger_paths ]


    def analyze(self, event):
        # Define Physics Objects
        electrons = Collection(event, 'Electron')
        trig_L1   = Object(event, 'L1')
        trig_HLT  = Object(event, 'HLT')
        pv        = Object(event, 'PV')

        # Define Trigger Paths
        path_1     = trig_L1.DoubleEG11_er1p2_dR_Max0p6   and trig_HLT.DoubleEle6p5_eta1p22_mMax6
        path_2     = trig_L1.DoubleEG10p5_er1p2_dR_Max0p6 and trig_HLT.DoubleEle6p5_eta1p22_mMax6
        path_3     = trig_L1.DoubleEG10p5_er1p2_dR_Max0p6 and trig_HLT.DoubleEle5_eta1p22_mMax6
        path_4     = trig_L1.DoubleEG9_er1p2_dR_Max0p7    and trig_HLT.DoubleEle6_eta1p22_mMax6
        path_5     = trig_L1.DoubleEG8p5_er1p2_dR_Max0p7  and trig_HLT.DoubleEle5p5_eta1p22_mMax6
        path_6     = trig_L1.DoubleEG8p5_er1p2_dR_Max0p7  and trig_HLT.DoubleEle5_eta1p22_mMax6
        path_7     = trig_L1.DoubleEG8_er1p2_dR_Max0p7    and trig_HLT.DoubleEle5_eta1p22_mMax6
        path_8     = trig_L1.DoubleEG7p5_er1p2_dR_Max0p7  and trig_HLT.DoubleEle5_eta1p22_mMax6
        path_9     = trig_L1.DoubleEG7_er1p2_dR_Max0p8    and trig_HLT.DoubleEle5_eta1p22_mMax6
        path_10    = trig_L1.DoubleEG6p5_er1p2_dR_Max0p8  and trig_HLT.DoubleEle4p5_eta1p22_mMax6
        path_11    = trig_L1.DoubleEG6_er1p2_dR_Max0p8    and trig_HLT.DoubleEle4_eta1p22_mMax6
        path_12    = trig_L1.DoubleEG5p5_er1p2_dR_Max0p8  and trig_HLT.DoubleEle6_eta1p22_mMax6
        path_13    = trig_L1.DoubleEG5p5_er1p2_dR_Max0p8  and trig_HLT.DoubleEle4_eta1p22_mMax6
        path_14    = trig_L1.DoubleEG5_er1p2_dR_Max0p9    and trig_HLT.DoubleEle4_eta1p22_mMax6
        path_15    = trig_L1.DoubleEG4p5_er1p2_dR_Max0p9  and trig_HLT.DoubleEle4_eta1p22_mMax6
        path_or    = path_1 or path_2 or path_3 or path_4 or path_5 or path_6 or path_7 or path_8 \
                     or path_9 or path_10 or path_11 or path_12 or path_13 or path_14 or path_15
        trig_paths = [path_or, path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8,
                      path_9, path_10, path_11, path_12, path_13, path_14, path_15]

        # Cuts corresponding to each trigger path
        dr_cuts         = [0.9, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9]
        sublead_pt_cuts = [4.5, 11., 10.5, 10.5, 9., 8.5, 8.5, 8., 7.5, 7, 6.5, 6, 5.5, 5.5, 5, 4.5]

        # Define Kinematic Variables
        lead_el_pt    = electrons[0].pt
        sublead_el_pt = electrons[1].pt
        sublead_eta   = electrons[1].eta
        sublead_phi   = electrons[1].phi
        dr            = electrons[0].DeltaR(electrons[1])
        npv           = pv.npvs
        diel_pt       = (electrons[0].p4() + electrons[1].p4()).Pt()
        diel_m        = (electrons[0].p4() + electrons[1].p4()).M()

        # Fill Kinematic Plots
        self.h_lead_el_pt.Fill(lead_el_pt)
        self.h_sublead_el_pt.Fill(sublead_el_pt)
        self.h_sublead_el_eta.Fill(sublead_eta)
        self.h_sublead_el_phi.Fill(sublead_phi)
        self.h_dr.Fill(dr)
        self.h_diel_m.Fill(diel_m)
        self.h_npv.Fill(npv)
        self.h_diel_pt.Fill(diel_pt)
        self.h2_sublead_el_pt_dr.Fill(sublead_el_pt,dr)

        for electron in electrons:
            self.h_el_pt.Fill(electron.pt)
            self.h_el_eta.Fill(electron.eta)
            self.h_el_phi.Fill(electron.phi)

        if path_or:
            self.h_sublead_el_pt_trigger_or.Fill(sublead_el_pt)
            self.h_sublead_el_eta_trigger_or.Fill(sublead_eta)
            self.h_sublead_el_phi_trigger_or.Fill(sublead_phi)
            self.h_dr_trigger_or.Fill(dr)
            self.h_diel_m_trigger_or.Fill(diel_m)
            self.h_npv_trigger_or.Fill(npv)
            self.h2_sublead_el_pt_dr_trigger_or.Fill(sublead_el_pt,dr)
            for electron in electrons:
                self.h_el_pt_trigger_or.Fill(electron.pt)
                self.h_el_eta_trigger_or.Fill(electron.eta)
                self.h_el_phi_trigger_or.Fill(electron.phi)

        # MC Efficiencies
        if self.isMC:
            # Assign MC events to trigger paths according to fractional Lint
            trig_Lints = [1.5774997, 1.1357226, 0.1028168, 8.8440658, 3.3395851, 0.6748351, 6.8900702, 1.6358989, 2.6620958, 3.6112524, 2.5114598, 0.1495199, 0.6480399, 0.0413131, 0.0299539]
            trig_idx = assign_frac_trig(event._entry, trig_Lints, event._tree.GetEntries())
            idx = trig_idx + 1
            if idx > len(trig_Lints): return True
        
            dr_cut = dr < dr_cuts[idx]
            sublead_pt_cut = sublead_el_pt > sublead_pt_cuts[idx]
            trig_cut = trig_paths[idx]

            # Trigger path PU dependence
            if trig_cut:
                self.h_npvs[0].Fill(npv)
                self.h_npvs[idx].Fill(npv)

            # Trigger Efficiencies
            # - Pt Eff (Apply dr and implicit eta cut)
            # - DR Eff (apply pt and implicit eta cut)
            # - Eta Eff (Apply dr and pt cuts)
            # - NPV Eff (apply pt, dr, and implicit eta cut)
            # - Di-E Pt Eff (apply pt, dr, and implicit eta cut)
            
            # Choose 'loose' or 'tight' method for combining paths to make 'trigger_OR'
            trigger_or_method = 'tight'
            if 'loose' in trigger_or_method:
                incl_dr_cut = dr < dr_cuts[0]
                incl_sublead_pt_cut = sublead_el_pt > sublead_pt_cuts[0]
                incl_trig_cut = np.logical_or.reduce(trig_paths[1:trig_idx+1])
                if incl_dr_cut:
                    self.h_diel_m_denom_ptbinned[0].Fill(diel_m,sublead_el_pt)
                    if incl_trig_cut: self.h_diel_m_num_ptbinned[0].Fill(diel_m,sublead_el_pt)
                if incl_sublead_pt_cut:
                    self.h_diel_m_denom_drbinned[0].Fill(diel_m,dr)
                    if incl_trig_cut: self.h_diel_m_num_drbinned[0].Fill(diel_m,dr)
                if incl_dr_cut and incl_sublead_pt_cut:
                    self.h_diel_m_denom_etabinned[0].Fill(diel_m,sublead_eta)
                    self.h_diel_m_denom_npvbinned[0].Fill(diel_m,npv)
                    self.h_diel_m_denom_dieptbinned[0].Fill(diel_m,diel_pt)
                    if incl_trig_cut: 
                        self.h_diel_m_num_etabinned[0].Fill(diel_m,sublead_eta)
                        self.h_diel_m_num_npvbinned[0].Fill(diel_m,npv)
                        self.h_diel_m_num_dieptbinned[0].Fill(diel_m,diel_pt)
            elif 'tight' in trigger_or_method:
                if dr_cut:
                    self.h_diel_m_denom_ptbinned[0].Fill(diel_m,sublead_el_pt)
                    if trig_cut: self.h_diel_m_num_ptbinned[0].Fill(diel_m,sublead_el_pt)
                if sublead_pt_cut: 
                    self.h_diel_m_denom_drbinned[0].Fill(diel_m,dr)
                    if trig_cut: self.h_diel_m_num_drbinned[0].Fill(diel_m,dr)
                if dr_cut and sublead_pt_cut: 
                    self.h_diel_m_denom_etabinned[0].Fill(diel_m,sublead_eta)
                    self.h_diel_m_denom_npvbinned[0].Fill(diel_m,npv)
                    self.h_diel_m_denom_dieptbinned[0].Fill(diel_m,diel_pt)
                    if trig_cut: 
                        self.h_diel_m_num_etabinned[0].Fill(diel_m,sublead_eta)
                        self.h_diel_m_num_npvbinned[0].Fill(diel_m,npv)
                        self.h_diel_m_num_dieptbinned[0].Fill(diel_m,diel_pt)
            else: raise KeyError('Choose Allowed Method For Combining Trigger Paths')

            # Individual Trigger Path Effs
            if dr_cut:
                self.h_diel_m_denom_ptbinned[idx].Fill(diel_m,sublead_el_pt)
                if trig_cut: self.h_diel_m_num_ptbinned[idx].Fill(diel_m,sublead_el_pt)
            if sublead_pt_cut: 
                self.h_diel_m_denom_drbinned[idx].Fill(diel_m,dr)
                if trig_cut: self.h_diel_m_num_drbinned[idx].Fill(diel_m,dr)
            if dr_cut and sublead_pt_cut: 
                self.h_diel_m_denom_etabinned[idx].Fill(diel_m,sublead_eta)
                self.h_diel_m_denom_npvbinned[idx].Fill(diel_m,npv)
                self.h_diel_m_denom_dieptbinned[idx].Fill(diel_m,diel_pt)
                if trig_cut: 
                    self.h_diel_m_num_etabinned[idx].Fill(diel_m,sublead_eta)
                    self.h_diel_m_num_npvbinned[idx].Fill(diel_m,npv)
                    self.h_diel_m_num_dieptbinned[idx].Fill(diel_m,diel_pt)

        else:
            # Trigger path PU dependence
            for hist, dr_cut, sublead_pt_cut, trig_cut in zip(self.h_npvs, dr_cuts, sublead_pt_cuts, trig_paths):
                if dr < dr_cut and sublead_el_pt > sublead_pt_cut and trig_cut: hist.Fill(npv)

            # Pt Eff (Apply dr and implicit eta cut)
            for num_hist, denom_hist, dr_cut, trig_cut in zip(self.h_diel_m_num_ptbinned, self.h_diel_m_denom_ptbinned, dr_cuts, trig_paths):
                if dr < dr_cut: 
                    denom_hist.Fill(diel_m,sublead_el_pt)
                    if trig_cut:
                        num_hist.Fill(diel_m,sublead_el_pt)

            # Eta Eff (Apply dr and pt cuts)
            for num_hist, denom_hist, dr_cut, sublead_pt_cut, trig_cut in zip(self.h_diel_m_num_etabinned, self.h_diel_m_denom_etabinned, dr_cuts, sublead_pt_cuts, trig_paths):
                if dr < dr_cut and sublead_el_pt > sublead_pt_cut: 
                    denom_hist.Fill(diel_m,sublead_eta)
                    if trig_cut: num_hist.Fill(diel_m,sublead_eta)
            
            # DR Eff (apply pt and implicit eta cut)
            for num_hist, denom_hist, sublead_pt_cut, trig_cut in zip(self.h_diel_m_num_drbinned, self.h_diel_m_denom_drbinned, sublead_pt_cuts, trig_paths):
                if sublead_el_pt > sublead_pt_cut: 
                    denom_hist.Fill(diel_m,dr)
                    if trig_cut: num_hist.Fill(diel_m,dr)

            # NPV Eff (apply pt, dr, and implicit eta cut)
            for num_hist, denom_hist, dr_cut, sublead_pt_cut, trig_cut in zip(self.h_diel_m_num_npvbinned, self.h_diel_m_denom_npvbinned, dr_cuts, sublead_pt_cuts, trig_paths):
                if dr < dr_cut and sublead_el_pt > sublead_pt_cut: 
                    denom_hist.Fill(diel_m,npv)
                    if trig_cut: 
                        num_hist.Fill(diel_m,npv)

            # Di-E Pt Eff (apply pt, dr, and implicit eta cut)
            for num_hist, denom_hist, dr_cut, sublead_pt_cut, trig_cut in zip(self.h_diel_m_num_dieptbinned, self.h_diel_m_denom_dieptbinned, dr_cuts, sublead_pt_cuts, trig_paths):
                if dr < dr_cut and sublead_el_pt > sublead_pt_cut: 
                    denom_hist.Fill(diel_m,diel_pt)
                    if trig_cut: num_hist.Fill(diel_m,diel_pt)

        return True

def worker(params):
    p = PostProcessor(
            params['output_dir'], 
            glob.glob(params['input_files']),
            cut=params['presel'],
            branchsel=None, 
            modules=[TriggerEfficiencyProducer(trigs, isMC=False if '2022' in params['name'] else True)],
            noOut=True,
            histFileName=params['output_dir']+'/'+params['output_filename'],
            histDirName='plots',
            jsonInput=params['json'],
    )
    p.run()

if __name__=='__main__':
    input_config = [
        # {'name' : 'Test',
        #  'input_files' : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022E/Preselection/output_22*_*_skim.root',
        #  'output_dir'  : '/afs/cern.ch/work/n/nzipper/public/Rk/TriggerEfficiencies/NanoAODTools/root_output/test',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : None,
        # },
        # {'name'        : 'BuToKee',
        #  'input_files' : '', 
        #  'output_dir'  : '/afs/cern.ch/work/n/nzipper/public/Rk/TriggerEfficiencies/NanoAODTools/root_output/BuToKee',
        #  'json_path'   : None,
        #  'presel'      : None,
        # },
        # {'name'        : 'BuToKJpsi_Toee_baseline_Tight',
        #  'input_files' : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/BuTOjpsiKEESkims/*.root', 
        #  'output_dir'  : '/afs/cern.ch/work/n/nzipper/public/Rk/TriggerEfficiencies/NanoAODTools/root_output/mc/BuToKJpsi_Toee',
        #  'json_path'   : None,
        #  'presel'      : None,
        # },
        # {'name'        : '2022C',
        #  'input_files' : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022C/Preselection/*.root',
        #  'output_dir'  : '/afs/cern.ch/work/n/nzipper/public/Rk/TriggerEfficiencies/NanoAODTools/root_output/data/data_baseline_excl',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : None,
        # },
        # {'name'        : '2022D',
        #  'input_files' : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022D/Preselection/*.root',
        #  'output_dir'  : '/afs/cern.ch/work/n/nzipper/public/Rk/TriggerEfficiencies/NanoAODTools/root_output/data/data_baseline_excl',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : None,
        # },
        # {'name'        : '2022E',
        #  'input_files' : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022E/Preselection/*.root',
        #  'output_dir'  : '/afs/cern.ch/work/n/nzipper/public/Rk/TriggerEfficiencies/NanoAODTools/root_output/data/data_baseline_excl',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : None,
        # },
        {'name'        : '2022F',
         'input_files' : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022F/Preselection/*.root',
         'output_dir'  : '/afs/cern.ch/work/n/nzipper/public/Rk/TriggerEfficiencies/NanoAODTools/root_output/data/data_baseline_excl',
         'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
         'presel'      : None,
        },
        # {'name'        : '2022G',
        #  'input_files' : '/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/ParkingDoubleMuonLowMassSkims/2022G/Preselection/*.root',
        #  'output_dir'  : '/afs/cern.ch/work/n/nzipper/public/Rk/TriggerEfficiencies/NanoAODTools/root_output/data/data_baseline_excl',
        #  'json_path'   : '/afs/cern.ch/work/n/nzipper/public/Rk/Files/JSON/Eras_CDEFG',
        #  'presel'      : None,
        # },
    ]

    trigs = [
        'trigger_OR',
        # 'L1_11p0_HLT_6p5_Incl_Final',
        # 'L1_10p5_HLT_6p5_Incl_Final',
        # 'L1_10p5_HLT_5p0_Incl_Final',
        # 'L1_9p0_HLT_6p0_Incl_Final',
        # 'L1_8p5_HLT_5p5_Incl_Final',
        # 'L1_8p5_HLT_5p0_Incl_Final',
        # 'L1_8p0_HLT_5p0_Incl_Final',
        # 'L1_7p5_HLT_5p0_Incl_Final',
        # 'L1_7p0_HLT_5p0_Incl_Final',
        # 'L1_6p5_HLT_4p5_Incl_Final',
        # 'L1_6p0_HLT_4p0_Incl_Final',
        # 'L1_5p5_HLT_6p0_Incl_Final',
        # 'L1_5p5_HLT_4p0_Incl_Final',
        # 'L1_5p0_HLT_4p0_Incl_Final',
        # 'L1_4p5_HLT_4p0_Incl_Final',
        'L1_11p0_HLT_6p5_Excl_Final',
        'L1_10p5_HLT_6p5_Excl_Final',
        'L1_10p5_HLT_5p0_Excl_Final',
        'L1_9p0_HLT_6p0_Excl_Final',
        'L1_8p5_HLT_5p5_Excl_Final',
        'L1_8p5_HLT_5p0_Excl_Final',
        'L1_8p0_HLT_5p0_Excl_Final',
        'L1_7p5_HLT_5p0_Excl_Final',
        'L1_7p0_HLT_5p0_Excl_Final',
        'L1_6p5_HLT_4p5_Excl_Final',
        'L1_6p0_HLT_4p0_Excl_Final',
        'L1_5p5_HLT_6p0_Excl_Final',
        'L1_5p5_HLT_4p0_Excl_Final',
        'L1_5p0_HLT_4p0_Excl_Final',
        'L1_4p5_HLT_4p0_Excl_Final',
        ]

    runStrategy = 'mp'
    if 'mp' in runStrategy:
        start_time = time.perf_counter()
        jobs = []
        for params in input_config:
            os.makedirs(params['output_dir'], exist_ok=True)
            if params['json_path']: 
                for trig in trigs:
                    params['json'] = ''.join([params['json_path'], '/', trig, '.json'])
                    params['output_filename'] = ''.join(['Efficiencies_', params['name'], '_', trig, '.root'])
                    pool = multiprocessing.Pool()
                    proc = multiprocessing.Process(target=worker, args=(params, ))
                    jobs.append(proc)
                    proc.start()
            else:
                params['json'] = None
                params['output_filename'] = ''.join(['Efficiencies_', params['name'], '.root'])
                pool = multiprocessing.Pool()
                proc = multiprocessing.Process(target=worker, args=(params, ))
                jobs.append(proc)
                proc.start()

        for p in jobs:
            p.join()

        finish_time = time.perf_counter()
        print(''.join(['Finished in ', str(finish_time-start_time), ' seconds']))

    else:
        for params in input_config:
            os.makedirs(params['output_dir'], exist_ok=True)
            if params['json_path']: 
                for trig in trigs:
                    params['json'] = ''.join([params['json_path'], '/', trig, '.json'])
                    params['output_filename'] = ''.join(['Efficiencies_', params['name'], '_', trig, '.root'])
                    worker(params)
            else:
                params['json'] = None
                params['output_filename'] = ''.join(['Efficiencies_', params['name'], '.root'])
                worker(params)