import os
import numpy as np
import json
import math
import pandas as pd
from tqdm import tqdm
import pickle
from glob import glob
from root_plotting import *

import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kWarning

get_ipython().run_cell_magic('cpp', '', '#include "RooDoubleCB.cc"\n')


def getCountedEvents(h, xrange=None):
    nEvents = 0
    err = np.double(0.)
    if xrange is not None:
        bin_low = h.GetXaxis().FindBin(xrange[0])
        bin_high = h.GetXaxis().FindBin(xrange[1])
        nEvents = h.IntegralAndError(bin_low, bin_high, err)
    else:
        nEvents = h.GetEntries()
        err = 1 / np.sqrt(nEvents)
    return nEvents, err

def estimateBins(h,nbins=5):
    evts = [h.GetBinContent(i) for i in range(h.GetXaxis().GetNbins())]
    tot = np.sum(evts)

    frac = 1
    n_evts = 0
    bins = f'{h.GetBinLowEdge(1)}'
    for i in range(1, h.GetXaxis().GetNbins()):
        n_evts += h.GetBinContent(i)
        if n_evts > frac * tot / nbins: 
            bins += ', '+str(round(h.GetBinLowEdge(i),2))
            frac += 1
    bins += ', '+str(round(h.GetBinLowEdge(h.GetXaxis().GetNbins()+1),2))
    return bins

def rejectFit(counts, err, chisqr=None, isMC=False):
    out = False
    if isMC:
        if counts < 1: out = True 
        if chisqr is not None:
            if chisqr>10: out = True
    else:
        if counts < 10: out = True 
        if counts < err: out = True 
        if chisqr is not None:
            if chisqr>10: out = True
    return out

def print_badfits(list):
    print(f'{"Trigger Path" : <25}{"Binning": <10}{"Bin" : <5}{"Fraction" : <10}{"Chi-Square Value"}')
    print(f'{"~"*24} {"~"*9} {"~"*4} {"~"*9} {"~"*16}')
    for fit_dict in list:
        frac = 'Num' if fit_dict['frac']=='num' else 'Denom'
        print(f'{fit_dict["trigger"]: <25}{fit_dict["key"]: <10}{fit_dict["bin"]: <5}{frac: <10}{round(fit_dict["chi square"],3)}')

def fit_jpsi_simple(h, signal_model='gauss', bkg_model='exp', signal_params=None, bkg_params=None, get_params=False, savename=None, show=False, verbose=False, get_chisqr=True, polypars=2):
    m_JPsi = 3.096900
    data_yield = h.GetEntries()
    data_min = h.GetXaxis().GetXmin()
    data_max = h.GetXaxis().GetXmax()

    mass = ROOT.RooRealVar('mass', 'mass', data_min, data_max)
    data = ROOT.RooDataHist('data', 'data', mass, h)
    if signal_model:
        if 'gauss' in signal_model:
            if signal_params:
                mean  = ROOT.RooRealVar('mean', 'mean', signal_params[0])
                sigma = ROOT.RooRealVar('sigma', 'sigma', signal_params[1])
            else:
                mean  = ROOT.RooRealVar('mean', 'mean', m_JPsi, 3.0 , 3.15)
                sigma = ROOT.RooRealVar('sigma', 'sigma', .03 , .02, .07)
            signal = ROOT.RooGaussian('signal', 'signal', mass, mean, sigma)
        elif 'dcb' in signal_model:
            mean = ROOT.RooRealVar('mean', 'mean', m_JPsi, m_JPsi-.05 , m_JPsi+.05)
            if signal_params:
                sigma = ROOT.RooRealVar('sigma', 'sigma', signal_params[1])
                al    = ROOT.RooRealVar('al', 'al', signal_params[2])
                ar    = ROOT.RooRealVar('ar', 'ar', signal_params[3])
                nl    = ROOT.RooRealVar('nl', 'nl', signal_params[4])
                nr    = ROOT.RooRealVar('nr', 'nr', signal_params[5])
                dcb   = ROOT.RooDoubleCB('dcb', 'dcb', mass, mean, sigma, al, ar, nl, nr)
            else:
                sigma = ROOT.RooRealVar('sigma', 'sigma', .03 , .01, .07)
                al = ROOT.RooRealVar('al', 'al', .2, 10.)
                ar = ROOT.RooRealVar('ar', 'ar', .2, 10.)
                nl = ROOT.RooRealVar('nl', 'nl', .2, 10.)
                nr = ROOT.RooRealVar('nr', 'nr', .2, 10.)

            signal = ROOT.RooDoubleCB('signal', 'signal', mass, mean, sigma, al, ar, nl, nr)
        else:
            raise ValueError('Input Valid Signal Model')
    if bkg_model:
        if 'exp' in bkg_model:
            if bkg_params:
                alpha = ROOT.RooRealVar('alpha', 'alpha', bkg_params[0], .5 * bkg_params, 2 * bkg_params)
            else:
                alpha = ROOT.RooRealVar('alpha', 'alpha', -1, -10, 0)
            bkg = ROOT.RooExponential('bkg','bkg',mass,alpha)    
        elif 'poly' in bkg_model:
            if bkg_params:
                pars = [ROOT.RooRealVar(f'a{i}', f'a{i}', par, .5 * par, 2 * par) for i, par in enumerate(bkg_params[:-1])]
                offset = ROOT.RooRealVar('offset','offset', bkg_params[-2], .5 * bkg_params[-2], 2 * bkg_params[-2])
            else:
                offset = ROOT.RooRealVar('offset','offset', m_JPsi, -10, 10)
                par_range = [0, -50,50]
                pars = [ROOT.RooRealVar(f'a{i}', f'a{i}', par_range[0], par_range[1], par_range[2]) for i in range(polypars)]

            diff = ROOT.RooFormulaVar('diff','mass-offset', ROOT.RooArgList(mass, offset))
            bkg = ROOT.RooPolynomial('bkg', 'bkg', diff, ROOT.RooArgList(*pars))
        else:
            raise ValueError('Input Valid Background Model')

    if signal_model and bkg_model: 
        sigyield = ROOT.RooRealVar('sigyield', 'signal yield in model', .01 * data_yield, 0, data_yield)
        bkgyield = ROOT.RooRealVar('bkgyield', 'background yield in model', .8 * data_yield, 0, data_yield)
        model = ROOT.RooAddPdf('model', 'model', ROOT.RooArgList(signal, bkg), ROOT.RooArgList(sigyield, bkgyield))
        r = model.fitTo(data, ROOT.RooFit.Save(True), ROOT.RooFit.Verbose(False), ROOT.RooFit.PrintEvalErrors(-1), ROOT.RooFit.PrintLevel(-1))
    elif signal_model: 
        sigyield = ROOT.RooRealVar('sigyield', 'signal yield in model', 0, data_yield)
        model = ROOT.RooAddPdf('model', 'model', ROOT.RooArgList(signal), ROOT.RooArgList(sigyield))
        r = model.fitTo(data, ROOT.RooFit.Save(True), ROOT.RooFit.Verbose(False), ROOT.RooFit.PrintEvalErrors(-1), ROOT.RooFit.PrintLevel(-1))
    elif bkg_model: 
        bkgyield = ROOT.RooRealVar('bkgyield', 'background yield in model', data_yield, 0, data_yield)        
        model = ROOT.RooAddPdf('model', 'model', ROOT.RooArgList(bkg), ROOT.RooArgList(bkgyield))

        mass.setRange("sb1", data_min, m_JPsi - .35)
        mass.setRange("sb2", m_JPsi + .35, data_max)
        # sb_data = data.reduce(ROOT.RooFit.CutRange("sb_1, sb_2"))
        r = model.fitTo(data, ROOT.RooFit.Range("sb1"), ROOT.RooFit.Save(True))#, ROOT.RooFit.Verbose(True), ROOT.RooFit.PrintEvalErrors(-1), ROOT.RooFit.PrintLevel(-1))
        # r1 = model.fitTo(data, ROOT.RooFit.Save(True))#, ROOT.RooFit.Verbose(True), ROOT.RooFit.PrintEvalErrors(-1), ROOT.RooFit.PrintLevel(-1))
        # r1.Print(0)
    else: raise ValueError('Input Some Type of Model')

    if signal_model and bkg_model: 
        out = [sigyield.getVal(), sigyield.getError()]
        if 'gauss' in signal_model:
            out_params = (mean.getVal(), sigma.getVal())
        elif 'dcb' in signal_model:
            out_params = (mean.getVal(), sigma.getVal(), al.getVal(), ar.getVal(), nl.getVal(), nr.getVal())
    elif signal_model: 
        out = [sigyield.getVal(), sigyield.getError()]
        if 'gauss' in signal_model:
            out_params = (mean.getVal(), sigma.getVal())
        elif 'dcb' in signal_model:
            out_params = (mean.getVal(), sigma.getVal(), al.getVal(), ar.getVal(), nl.getVal(), nr.getVal())
    elif bkg_model: 
        out = [bkgyield.getVal(), bkgyield.getError()]
        if 'exp' in bkg_model:
            out_params = (alpha.getVal())
        elif 'poly' in bkg_model:
            out_params = tuple(par.getVal() for par in pars)
            out_params = out_params + (offset.getVal(),)
    else: raise ValueError('Input Some Type of Model')

    if get_params: out.append(out_params)

    if show or savename:
        c = ROOT.TCanvas()
        frame = mass.frame(ROOT.RooFit.Title('Fit' if not savename else savename))
        data.plotOn(frame, ROOT.RooFit.Name('data'),  ROOT.RooFit.CutRange("sb_1, sb_2"))

        if signal_model and bkg_model:
            model.plotOn(frame, ROOT.RooFit.Name('model'), ROOT.RooFit.LineColor(38), ROOT.RooFit.Normalization(sigyield.getVal()+bkgyield.getVal(),ROOT.RooAbsReal.NumEvent))
            model.plotOn(frame, ROOT.RooFit.Components('signal'), ROOT.RooFit.LineColor(32))
            model.plotOn(frame, ROOT.RooFit.Components('bkg'), ROOT.RooFit.LineColor(46))
        elif signal_model:
            model.plotOn(frame, ROOT.RooFit.Name('model'), ROOT.RooFit.LineColor(38), ROOT.RooFit.Normalization(sigyield.getVal(),ROOT.RooAbsReal.NumEvent))
        elif bkg_model:
            model.plotOn(frame, ROOT.RooFit.Name('model'), ROOT.RooFit.LineColor(38), ROOT.RooFit.Normalization(bkgyield.getVal(),ROOT.RooAbsReal.NumEvent),ROOT.RooFit.Range("sb_1, sb_2"))
        else: raise ValueError('Input Some Type of Model')

        model.paramOn(frame, ROOT.RooFit.Layout(0.12, 0.3, 0.88), ROOT.RooFit.Format('NE', ROOT.RooFit.FixedPrecision(3)))

        frame.getAttText().SetTextSize(.025) 
        frame.Draw()

        text = ROOT.TText()
        text.SetTextFont(43)
        text.SetTextSize(20)
        chisqr = frame.chiSquare('model', 'data')
        # chisqr = ROOT.RooChi2Var('chi2', 'chi2', model, data).getVal()
        text.DrawTextNDC(.65, .85, f'Chi-Square = {round(chisqr,2)}')

        if get_chisqr: out.append(chisqr)

        # if verbose: 
        #     print(h.GetTitle(), chisqr)

        if savename: 
            # os.makedirs()
            # print(os.path.abspath(savename))
            c.SaveAs(savename)
        if show: out.append(c)

        # Recurse polynomial fit to improve convergence
        if bkg_model:
            if 'poly' in bkg_model:
                if verbose: print(f'Background Fit Order: {polypars} - Chi-Square: {chisqr}')
                if chisqr > 1. and polypars<9:
                        new_out = fit_jpsi_simple(h, signal_model, bkg_model, signal_params, bkg_params, get_params, savename, show, verbose, get_chisqr=True, polypars=polypars+1)
                        if new_out[3] < out[3]: out = new_out

        if show: c.Draw()
            
    return out


def processMCFitParams(filename, trigger_dict):
  print('Generating Fit Parameters...')
  ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.MsgLevel.ERROR)
  f = ROOT.TFile(filename)
  outputs_params = {}
  for j, (num_name, (num_path, denom_path)) in enumerate(pbar := tqdm(trigger_dict.items(), ncols=100, unit=' Triggers')):
    pbar.set_description(num_name)
    outputs_params_var = {}
    for name, opts in options.items():
      nbins = len(opts['bins']) - 1
      eff_output = np.zeros(nbins, np.dtype('<U99,float,float,float,float'))

      # TODO Remove Excl/Incl naming for MC in DiElectronEfficiencies.py script (lxplus)
      if False:
        if 'Excl' in num_path:
          num_path = num_path.replace('Excl','Incl')
          denom_path = denom_path.replace('Excl','Incl')
        elif 'Incl' in num_path:
          num_path = num_path.replace('Incl','Excl')
          denom_path = denom_path.replace('Incl','Excl')

      num_h = f.Get(''.join([num_path,opts['suffix']]))
      denom_h = f.Get(''.join([denom_path,opts['suffix']]))
      outputs_params_num_bin = []
      outputs_params_denom_bin = []
      for i in range(1, nbins + 1):
        # print(num_name, name, i)
        num_bin = num_h.ProjectionX('h1',i,i+1)
        denom_bin = denom_h.ProjectionX('h2',i,i+1)

        _, _, num_params = fit_jpsi_simple(num_bin, signal_model='dcb', bkg_model=None, signal_params=None, get_params=True, savename=None, show=False)
        _, _, denom_params = fit_jpsi_simple(denom_bin, signal_model='dcb', bkg_model=None, signal_params=None, get_params=True, savename=None, show=False)
        outputs_params_num_bin.append(num_params)
        outputs_params_denom_bin.append(denom_params)

      outputs_params_var[name] = {'num' : outputs_params_num_bin, 'denom' : outputs_params_denom_bin}
    outputs_params[num_name] = outputs_params_var

  return outputs_params


def processMCEffs(filename, trigger_dict, verbose=False, paramset=None, savefits=False):
  print('Calculating Efficiencies for MC...')
  ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.MsgLevel.ERROR)
  f = ROOT.TFile(filename)

  outputs_mc = []
  badfits_mc = []
  for j, (num_name, (num_path, denom_path)) in enumerate(pbar := tqdm(trigger_dict.items(), ncols=100, unit=' Triggers')):
    pbar.set_description(num_name)
    output_dict_mc = {}
    for name, opts in options.items():
      nbins = len(opts['bins']) - 1
      eff_output = np.zeros(nbins,np.dtype('<U99,float,float,float,float'))

      # TODO Remove Excl/Incl naming for MC in DiElectronEfficiencies.py script (lxplus)
      if False:
        if 'Excl' in num_path:
          num_path = num_path.replace('Excl','Incl')
          denom_path = denom_path.replace('Excl','Incl')
        elif 'Incl' in num_path:
          num_path = num_path.replace('Incl','Excl')
          denom_path = denom_path.replace('Incl','Excl')

      num_h = f.Get(''.join([num_path,opts['suffix']]))
      denom_h = f.Get(''.join([denom_path,opts['suffix']]))
      
      for i in range(1, nbins + 1):
        try: num_bin = num_h.ProjectionX('num_bin',i,i+1)
        except AttributeError('Cannot Find Numerator Histogram'): continue
        try: denom_bin = denom_h.ProjectionX('denom_bin',i,i+1)
        except AttributeError('Cannot Find Denominator Histogram'): continue

        num_counts, num_err, chisqr = fit_jpsi_simple(num_bin, 
                                            signal_model='dcb', 
                                            signal_params=paramset[num_name][name]['num'][i-1] if paramset else None, 
                                            bkg_model=None, 
                                            get_params=False, 
                                            get_chisqr=True,
                                            savename=f'plots/fits/mc/{name}/{num_name}_{name}binned_bin{i}_mc_num.png' if savefits else False)
        # if num_bin.GetEntries() < 3: num_counts, num_err = 0, 0
        if rejectFit(num_counts, num_err, chisqr, isMC=True): 
          num_counts, num_err = 0, 0
          badfits_mc.append({'trigger' : num_name, 'key': name, 'bin' : i, 'frac' : 'num', 'chi square' : chisqr})

        denom_counts, denom_err, chisqr = fit_jpsi_simple(denom_bin, 
                                                signal_model='dcb', 
                                                bkg_model=None, 
                                                signal_params=paramset[num_name][name]['denom'][i-1] if paramset else None, 
                                                get_params=False, 
                                                get_chisqr=True,
                                                savename=f'plots/fits/mc/{name}/{num_name}_{name}binned_bin{i}_mc_denom.png' if savefits else False)
        # if denom_bin.GetEntries() < 3: num_counts, num_err = 0, 0        
        if rejectFit(denom_counts, denom_err, chisqr, isMC=True):
          denom_counts, denom_err = 0, 0
          badfits_mc.append({'trigger' : num_name, 'key': name, 'bin' : i, 'frac' : 'den', 'chi square' : chisqr})

        eff_output[i-1] = (num_name,num_counts,num_err,denom_counts,denom_err)
      output_dict_mc[name] = eff_output
    outputs_mc.append(output_dict_mc)

  return outputs_mc, badfits_mc


# In[6]:


def processDataEffs(filenames, trigger_dict, verbose=False, paramset=None, savefits=False):
  print('Calculating Efficiencies for Data...')
  ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.MsgLevel.ERROR)
  files = filenames
  outputs_data = []
  badfits_data = []

  for j, (num_name, (num_path, denom_path)) in enumerate(pbar := tqdm(trigger_dict.items(), ncols=100, unit=' Triggers')):
    pbar.set_description(num_name)
    output_dict_data = {}
    for name, opts in options.items():
      nbins = len(opts['bins']) - 1
      eff_output = np.zeros(nbins, np.dtype('<U99,float,float,float,float'))
      for filename in files:
        if num_name not in filename: continue
        f = ROOT.TFile(filename)
        num_h = f.Get(''.join([num_path,opts['suffix']]))
        denom_h = f.Get(''.join([denom_path,opts['suffix']]))
        for i in range(1, nbins + 1):
          try: num_bin = num_h.ProjectionX('num_bin',i,i+1)
          except AttributeError('Cannot Find Numerator Histogram'): continue
          try: denom_bin = denom_h.ProjectionX('denom_bin',i,i+1)
          except AttributeError('Cannot Find Denominator Histogram'): continue
          
          bkg_shape = 'poly' if 'dr' in name else 'exp'
          num_counts, num_err, _, chisqr = fit_jpsi_simple(num_bin, 
                                                 signal_model='dcb', 
                                                 signal_params=paramset[num_name][name]['num'][i-1] if paramset else None, 
                                                 bkg_model=bkg_shape, 
                                                 get_params=True, 
                                                 get_chisqr=True,
                                                 savename=f'plots/fits/data/{name}/{num_name}_{name}binned_bin{i}_data_num.png' if savefits else False)
          if num_bin.GetEntries() < 10: num_counts, num_err = 0, 0
          if rejectFit(num_counts, num_err, chisqr): 
            num_counts, num_err = 0, 0
            badfits_data.append({'trigger' : num_name, 'key': name, 'bin' : i, 'frac' : 'num', 'chi square' : chisqr})

          denom_counts, denom_err, _, chisqr = fit_jpsi_simple(denom_bin, 
                                                     signal_model='dcb', 
                                                     signal_params=paramset[num_name][name]['denom'][i-1] if paramset else None, 
                                                     bkg_model=bkg_shape, 
                                                     get_params=True, 
                                                     get_chisqr=True,
                                                     savename=f'plots/fits/data/{name}/{num_name}_{name}binned_bin{i}_data_denom.png' if savefits else False)
          if denom_bin.GetEntries() < 10: num_counts, num_err = 0, 0
          if rejectFit(denom_counts, denom_err, chisqr):
            denom_counts, denom_err = 0, 0
            badfits_data.append({'trigger' : num_name, 'key': name, 'bin' : i, 'frac' : 'den', 'chi square' : chisqr})
            
          eff_output[i-1] = (num_name,num_counts,num_err,denom_counts,denom_err)

      output_dict_data[name] = eff_output
      # if np.any(np.array([tuple(eff_output[i-1])[1:] for i in range(1, nbins + 1)])): output_dict_data[name] = eff_output 
    outputs_data.append(output_dict_data)
    # if output_dict_data: outputs_data.append(output_dict_data)

  return outputs_data, badfits_data


ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.MsgLevel.WARNING)

options = {
    'pt'    : {'suffix' : '_ptbinned',    'bins' : np.array([5, 7, 9, 10, 11, 12, 13, 20], dtype=np.double),        'xlabel' : r'Sublead \ Electron \ p_{T} \ [GeV]'},
    # 'dr'    : {'suffix' : '_drbinned',    'bins' : np.array([0, 0.12, 0.2, 0.28, 0.44, 1.], dtype=np.double),       'xlabel' : r'\Delta R(e_{1},e_{2})'},
    # 'npv'   : {'suffix' : '_npvbinned',   'bins' : np.array([0, 22, 27,31, 36, 100], dtype=np.double),              'xlabel' : r'N_{PV}'},
    # 'eta'   : {'suffix' : '_etabinned',   'bins' : np.array([-1.22, -0.7, -.2, 0.2, .7, 1.22], dtype=np.double),    'xlabel' : r'Sublead \ Electron \ \eta'},
    # 'diept' : {'suffix' : '_dieptbinned', 'bins' : np.array([5, 10, 11, 12, 15, 20, 40, 70, 100], dtype=np.double), 'xlabel' : r' Di-Electron p_{T} \ [GeV]'},
}

trigger_dict = {
  'trigger_OR'           : ('plots/diel_m_trigger_OR_num',                 'plots/diel_m_trigger_OR_denom'),
  'L1_11p0_HLT_6p5_Incl' : ('plots/diel_m_L1_11p0_HLT_6p5_Incl_Final_num', 'plots/diel_m_L1_11p0_HLT_6p5_Incl_Final_denom'),
  'L1_10p5_HLT_6p5_Incl' : ('plots/diel_m_L1_10p5_HLT_6p5_Incl_Final_num', 'plots/diel_m_L1_10p5_HLT_6p5_Incl_Final_denom'),
  'L1_10p5_HLT_5p0_Incl' : ('plots/diel_m_L1_10p5_HLT_5p0_Incl_Final_num', 'plots/diel_m_L1_10p5_HLT_5p0_Incl_Final_denom'),
  'L1_9p0_HLT_6p0_Incl'  : ('plots/diel_m_L1_9p0_HLT_6p0_Incl_Final_num',  'plots/diel_m_L1_9p0_HLT_6p0_Incl_Final_denom'),
  'L1_8p5_HLT_5p5_Incl'  : ('plots/diel_m_L1_8p5_HLT_5p5_Incl_Final_num',  'plots/diel_m_L1_8p5_HLT_5p5_Incl_Final_denom'),
  'L1_8p5_HLT_5p0_Incl'  : ('plots/diel_m_L1_8p5_HLT_5p0_Incl_Final_num',  'plots/diel_m_L1_8p5_HLT_5p0_Incl_Final_denom'),
  'L1_8p0_HLT_5p0_Incl'  : ('plots/diel_m_L1_8p0_HLT_5p0_Incl_Final_num',  'plots/diel_m_L1_8p0_HLT_5p0_Incl_Final_denom'),
  'L1_7p5_HLT_5p0_Incl'  : ('plots/diel_m_L1_7p5_HLT_5p0_Incl_Final_num',  'plots/diel_m_L1_7p5_HLT_5p0_Incl_Final_denom'),
  'L1_7p0_HLT_5p0_Incl'  : ('plots/diel_m_L1_7p0_HLT_5p0_Incl_Final_num',  'plots/diel_m_L1_7p0_HLT_5p0_Incl_Final_denom'),
  'L1_6p5_HLT_4p5_Incl'  : ('plots/diel_m_L1_6p5_HLT_4p5_Incl_Final_num',  'plots/diel_m_L1_6p5_HLT_4p5_Incl_Final_denom'),
  'L1_6p0_HLT_4p0_Incl'  : ('plots/diel_m_L1_6p0_HLT_4p0_Incl_Final_num',  'plots/diel_m_L1_6p0_HLT_4p0_Incl_Final_denom'),
  'L1_5p5_HLT_6p0_Incl'  : ('plots/diel_m_L1_5p5_HLT_6p0_Incl_Final_num',  'plots/diel_m_L1_5p5_HLT_6p0_Incl_Final_denom'),
  'L1_5p5_HLT_4p0_Incl'  : ('plots/diel_m_L1_5p5_HLT_4p0_Incl_Final_num',  'plots/diel_m_L1_5p5_HLT_4p0_Incl_Final_denom'),
  'L1_5p0_HLT_4p0_Incl'  : ('plots/diel_m_L1_5p0_HLT_4p0_Incl_Final_num',  'plots/diel_m_L1_5p0_HLT_4p0_Incl_Final_denom'),
  'L1_4p5_HLT_4p0_Incl'  : ('plots/diel_m_L1_4p5_HLT_4p0_Incl_Final_num',  'plots/diel_m_L1_4p5_HLT_4p0_Incl_Final_denom'),

  'L1_11p0_HLT_6p5_Excl' : ('plots/diel_m_L1_11p0_HLT_6p5_Excl_Final_num', 'plots/diel_m_L1_11p0_HLT_6p5_Excl_Final_denom'),
  'L1_10p5_HLT_6p5_Excl' : ('plots/diel_m_L1_10p5_HLT_6p5_Excl_Final_num', 'plots/diel_m_L1_10p5_HLT_6p5_Excl_Final_denom'),
  'L1_10p5_HLT_5p0_Excl' : ('plots/diel_m_L1_10p5_HLT_5p0_Excl_Final_num', 'plots/diel_m_L1_10p5_HLT_5p0_Excl_Final_denom'),
  'L1_9p0_HLT_6p0_Excl'  : ('plots/diel_m_L1_9p0_HLT_6p0_Excl_Final_num',  'plots/diel_m_L1_9p0_HLT_6p0_Excl_Final_denom'),
  'L1_8p5_HLT_5p5_Excl'  : ('plots/diel_m_L1_8p5_HLT_5p5_Excl_Final_num',  'plots/diel_m_L1_8p5_HLT_5p5_Excl_Final_denom'),
  'L1_8p5_HLT_5p0_Excl'  : ('plots/diel_m_L1_8p5_HLT_5p0_Excl_Final_num',  'plots/diel_m_L1_8p5_HLT_5p0_Excl_Final_denom'),
  'L1_8p0_HLT_5p0_Excl'  : ('plots/diel_m_L1_8p0_HLT_5p0_Excl_Final_num',  'plots/diel_m_L1_8p0_HLT_5p0_Excl_Final_denom'),
  'L1_7p5_HLT_5p0_Excl'  : ('plots/diel_m_L1_7p5_HLT_5p0_Excl_Final_num',  'plots/diel_m_L1_7p5_HLT_5p0_Excl_Final_denom'),
  'L1_7p0_HLT_5p0_Excl'  : ('plots/diel_m_L1_7p0_HLT_5p0_Excl_Final_num',  'plots/diel_m_L1_7p0_HLT_5p0_Excl_Final_denom'),
  'L1_6p5_HLT_4p5_Excl'  : ('plots/diel_m_L1_6p5_HLT_4p5_Excl_Final_num',  'plots/diel_m_L1_6p5_HLT_4p5_Excl_Final_denom'),
  'L1_6p0_HLT_4p0_Excl'  : ('plots/diel_m_L1_6p0_HLT_4p0_Excl_Final_num',  'plots/diel_m_L1_6p0_HLT_4p0_Excl_Final_denom'),
  'L1_5p5_HLT_6p0_Excl'  : ('plots/diel_m_L1_5p5_HLT_6p0_Excl_Final_num',  'plots/diel_m_L1_5p5_HLT_6p0_Excl_Final_denom'),
  'L1_5p5_HLT_4p0_Excl'  : ('plots/diel_m_L1_5p5_HLT_4p0_Excl_Final_num',  'plots/diel_m_L1_5p5_HLT_4p0_Excl_Final_denom'),
  'L1_5p0_HLT_4p0_Excl'  : ('plots/diel_m_L1_5p0_HLT_4p0_Excl_Final_num',  'plots/diel_m_L1_5p0_HLT_4p0_Excl_Final_denom'),
  'L1_4p5_HLT_4p0_Excl'  : ('plots/diel_m_L1_4p5_HLT_4p0_Excl_Final_num',  'plots/diel_m_L1_4p5_HLT_4p0_Excl_Final_denom'),
}

data_pathname = 'root_files/data/baseline'
# mc_filename = 'root_files/mc/Efficiencies_BuToKJpsi_Toee_Loose.root'
# mc_filename = 'root_files/mc/Efficiencies_BuToKJpsi_Toee_baseline_Tight.root'
mc_filename = 'root_files/mc/Efficiencies_BuToKJpsi_Toee_Reweighted.root'

mc_params = processMCFitParams(mc_filename, trigger_dict)
outputs_mc, badfits_mc = processMCEffs(mc_filename, trigger_dict, verbose=True, paramset=mc_params, savefits=True)
outputs_data, badfits_data = processDataEffs(glob(data_pathname+'/*.root'), trigger_dict, verbose=False, paramset=mc_params, savefits=True)
print('Data Fits:', badfits_data)
print('MC Fits:', badfits_mc)


# ## Plotting

# ### Single Hist Plotter

# In[9]:


# import matplotlib.pyplot as plt
# # ROOT.TH1.SetDefaultSumw2()
# ROOT.gROOT.SetBatch(True)

# options = {
#     'pt' : {'suffix' : '_ptbin', 'nbins' : 7, 'bins' : np.array([5, 5.4, 6.2, 7.4, 9.6, 10.5, 12., 50.], dtype=np.double)  , 'xlabel' : r'Sublead Electron p_{T} [GeV]'},
#     'eta' : {'suffix' : '_etabin', 'nbins' : 5, 'bins' : np.array([-1.22, -0.7, -.2, 0.2, .7, 1.22], dtype=np.double), 'xlabel' : r'Sublead Electron \eta'},
#     'dr' : {'suffix' : '_drbin', 'nbins' : 4, 'bins' : np.array([0, 0.1, 0.3, 0.6, 0.8], dtype=np.double), 'xlabel' : r'\Delta R(e_{1},e_{2})'},
#     'npv' : {'suffix' : '_npvbin', 'nbins' : 2, 'bins' : np.array([0, 35, 100], dtype=np.double), 'xlabel' : r'N_{PV}'},
#     'diept' : {'suffix' : '_dieptbin', 'nbins' : 7, 'bins' : np.array([5, 5.4, 6.2, 7.4, 9.6, 10.5, 12., 50.], dtype=np.double), 'xlabel' : r'Di-Electron p_{T} [GeV]'},
#   }
  
# usedata = True
# outputs = outputs_data if usedata else outputs_mc
# key = 'pt'
# overlap = False

# if overlap:
#     plt.figure(figsize=(10,8))
#     plt.title('Di-Electron Trigger Turn-On Curves')
#     plt.ylabel('Efficiency',loc='top')
#     plt.xlabel(xlabel,loc='right')
#     plt.ylim(0,1.)
#     plt.xlim(options[key]['bins'][0],options[key]['bins'][-1])
    

# for output in outputs:
#     eff_output = output[key]
#     bins = options[key]['bins']
#     xlabel = options[key]['xlabel']

#     for i in range(1):
#         xs = (bins[1:] + bins[:-1]) / 2
#         x_err=np.diff(bins)/2
        
#         title, nums, num_errs, denoms, denom_errs = np.array([[*i] for i in eff_output]).T
#         title = np.asarray(title)[np.nonzero(np.asarray(title))][0] if len(np.asarray(title)[np.nonzero(np.asarray(title))]) else ''
#         hpass = ROOT.TH1F('hpass', 'hpass', len(bins)-1, bins)
#         hall = ROOT.TH1F('hall', 'hall', len(bins)-1, bins)

#         nums = nums.astype(np.double)
#         num_errs = num_errs.astype(np.double)
#         denoms = denoms.astype(np.double)
#         denom_errs = denom_errs.astype(np.double)

#         for ibin, (num, num_err, denom, denom_err) in enumerate(zip(nums, num_errs, denoms, denom_errs)):
#             if num > denom: continue
#             hpass.SetBinContent(ibin+1,num)
#             hpass.SetBinError(ibin+1,num_err)
#             hall.SetBinContent(ibin+1,denom)
#             hall.SetBinError(ibin+1,denom_err)

#         eff = ROOT.TEfficiency(hpass, hall)
#         del hpass
#         del hall

#         ys = []
#         y_uerr = []
#         y_derr = []
#         for ibin in range(len(bins)-1):
#             ys.append(eff.GetEfficiency(ibin+1))
#             y_uerr.append(eff.GetEfficiencyErrorUp(ibin+1))
#             y_derr.append(eff.GetEfficiencyErrorLow(ibin+1))
#         y_err = [y_uerr,y_derr]

#         if not overlap:
#             eff.SetTitle(f'{title}; {xlabel}; Efficiency')
#             c = ROOT.TCanvas('c', 'c', 800,800)
#             eff.Draw()
#             ROOT.gPad.Update()
#             g = eff.GetPaintedGraph()
#             g.GetXaxis().SetLimits(options[key]['bins'][0],options[key]['bins'][-1])
#             g.GetHistogram().SetMinimum(0)
#             g.GetHistogram().SetMaximum(1)
#             g.Draw('E')
#             ROOT.gPad.Update()
#             c.SaveAs(f'plots/eff_{key}_{title}.pdf') if usedata else c.SaveAs(f'plots/eff_{key}_{title}_mc.pdf')
#             del c

#             plt.figure(figsize=(6,6))
#             plt.title(title)
#             plt.ylabel('Efficiency',loc='top')
#             plt.xlabel(xlabel,loc='right')
#             plt.ylim(0,1.)
#             plt.xlim(options[key]['bins'][0],options[key]['bins'][-1])
        

#         plt.errorbar(xs,ys,xerr=x_err,yerr=y_err,capsize=2,linestyle='none',label=title)

# if overlap: plt.legend(loc=1)


# ### Comparison Plotter

import matplotlib.pyplot as plt
from root_plotting.EfficiencyPlot import EfficiencyPlot
# import root_plotting 
ROOT.gROOT.ProcessLine('gErrorIgnoreLevel = kError;')

def saveEffsFromArrays(name, data, key, bins, effs, errup, errdown):
    filename = '_'.join(['Trigger_Effs/TrigEffs', 'Data' if data else 'MC', 'Incl', key+'binned.json'])
    
    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        with open(filename, 'r') as f:
            out_dict = json.load(f)
    else: out_dict = {}

    if name not in out_dict.keys(): out_dict[name] = {}

    out_dict[name][key] = bins[:-1].tolist()
    out_dict[name].update({'effs' : [(round(eff,3),round(err,3)) for eff,err in zip(effs,np.maximum(errup, errdown).tolist())]})

    with open(filename, 'w+') as f:
        json.dump(out_dict, f, indent=4)
        
def get_subplots_grid(i):
    if i==1: return 1,1
    for j in range(1,i):
        if i%j==0 and i/j<9 and i/j>1:
            return int(i/j), j
    return get_subplots_grid(i+1)

def plotEffComparison(dict_1, dict_2, options, key='pt', justIncl=False, show=True, labels=None, save=False):
    RooEff = EfficiencyPlot()
    label1, label2 = ('Data', 'MC') if labels is None else labels
    bins = options[key]['bins']
    xlabel = f'${options[key]["xlabel"]}$'
    nplots = 1 if justIncl or (len(dict_1)==1 and len(dict_2)==1) else len(dict_1)

    if show:
        if nplots==1:
            fig, axs = plt.subplots(figsize=(8,8))
        else:
            if len(dict_1) != len(dict_2): raise ValueError('Make sure Data and MC effs Match')
            nplots = len(dict_1)
            cols, rows  = get_subplots_grid(nplots)
            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            fig.text(.5, 0, xlabel, ha='center', va='center', fontsize=18)
            fig.text(.09, .6, 'Efficiency', va='top', rotation='vertical', fontsize=18)
            if cols*rows%nplots!=0: axs.ravel()[-1].axis("off")

    for i, (output_1, output_2) in enumerate(zip(dict_1, dict_2)):
        eff_output_1 = output_1[key]
        eff_output_2 = output_2[key]

        xs = (bins[1:] + bins[:-1]) / 2
        x_err=np.diff(bins)/2
        
        title, nums_1, num_errs_1, denoms_1, denom_errs_1 = np.array([[*i] for i in eff_output_1]).T
        title, nums_2, num_errs_2, denoms_2, denom_errs_2 = np.array([[*i] for i in eff_output_2]).T
        title = np.asarray(title)[np.nonzero(np.asarray(title))][0] if len(np.asarray(title)[np.nonzero(np.asarray(title))]) else ''
        if justIncl and 'trigger_OR' not in title: continue
        
        # # FIXME
        # title = title.replace('Incl', 'Excl')

        hpass_1 = ROOT.TH1F('hpass_1', 'hpass_1', len(bins)-1, bins)
        hpass_2 = ROOT.TH1F('hpass_2', 'hpass_2', len(bins)-1, bins)
        hall_1 = ROOT.TH1F('hall_1', 'hall_1', len(bins)-1, bins)
        hall_2 = ROOT.TH1F('hall_2', 'hall_2', len(bins)-1, bins)
        
        nums_1 = nums_1.astype(np.double)
        num_errs_1 = num_errs_1.astype(np.double)
        denoms_1 = denoms_1.astype(np.double)
        denom_errs_1 = denom_errs_1.astype(np.double)

        nums_2 = nums_2.astype(np.double)
        num_errs_2 = num_errs_2.astype(np.double)
        denoms_2 = denoms_2.astype(np.double)
        denom_errs_2 = denom_errs_2.astype(np.double)

        for ibin, (num, num_err, denom, denom_err) in enumerate(zip(nums_1, num_errs_1, denoms_1, denom_errs_1)):
            if num > denom: continue
            hpass_1.SetBinContent(ibin+1,num)
            hall_1.SetBinContent(ibin+1,denom)
            hpass_1.SetBinError(ibin+1,num_err)
            hall_1.SetBinError(ibin+1,denom_err)

        for ibin, (num, num_err, denom, denom_err) in enumerate(zip(nums_2, num_errs_2, denoms_2, denom_errs_2)):
            if num > denom: continue
            hpass_2.SetBinContent(ibin+1,num)
            hall_2.SetBinContent(ibin+1,denom)
            hpass_2.SetBinError(ibin+1,num_err)
            hall_2.SetBinError(ibin+1,denom_err)

        eff_1 = ROOT.TEfficiency(hpass_1, hall_1)
        eff_1.SetStatisticOption(ROOT.TEfficiency.kBBayesian)
        eff_2 = ROOT.TEfficiency(hpass_2, hall_2)
        eff_2.SetStatisticOption(ROOT.TEfficiency.kBBayesian)

        # ROOT Plot
        RooEff.set_params({'title_string' : f'{title};{options[key]["xlabel"]};Efficiency', 
                        #    'canvas_size' : (800,800),
                           'xrange' : (bins[0],bins[-1]),
                           'yrange' : (0.,1.1),
                        #    'rrange' : (.7,1.3) if justIncl else ((0,2) if 'dr' in key else (.2,3)),
        })
        a = RooEff.plotEfficiencies(eff_1, eff_2, ratio=True, h1_title='Data', h2_title='MC', save=f'plots/eff_{key}_{title}_comparison.png', addIntegral=False)
        del hpass_1, hall_1, hpass_2, hall_2

        # Show matplotlib panel
        if show:
            ys_1 = [eff_1.GetEfficiency(ibin) for ibin in range(1,len(bins))]
            ys_2 = [eff_2.GetEfficiency(ibin) for ibin in range(1,len(bins))]
            y_err_1 = np.array([[eff_1.GetEfficiencyErrorUp(ibin), eff_1.GetEfficiencyErrorLow(ibin)] for ibin in range(1,len(bins))]).T
            y_err_2 = np.array([[eff_2.GetEfficiencyErrorUp(ibin), eff_2.GetEfficiencyErrorLow(ibin)] for ibin in range(1,len(bins))]).T

            ax = axs if nplots==1 else axs.ravel()[i]
            if nplots==1: 
                ax.set_ylabel('Efficiency',loc='top')
                ax.set_xlabel(xlabel,loc='right')

            ax.set_title(title)
            ax.set_ylim(0,1.)
            ax.set_xlim(options[key]['bins'][0],options[key]['bins'][-1])
            
            ax.errorbar(xs,ys_1,xerr=x_err, yerr=y_err_1, capsize=2,linestyle='none',label=label1)
            ax.errorbar(xs,ys_2,xerr=x_err, yerr=y_err_2, capsize=2,linestyle='none',label=label2)
            ax.legend()
        
        # Save JSON
        if save:
            saveEffsFromArrays(title, True, key, bins, ys_1, y_err_1[0], y_err_1[1])
            saveEffsFromArrays(title, False, key, bins, ys_2, y_err_2[0], y_err_2[1])


# In[16]:


# Use cached fits
# with open('effs.pkl','rb') as f: mc_params, outputs_mc, outputs_data = pickle.load(f)

plotEffComparison(outputs_data, outputs_mc, options, key='pt', justIncl=False, show=True, save=True)
# plotEffComparison(outputs_data, outputs_mc, options, key='dr', justIncl=False, show=True, save=True)
# plotEffComparison(outputs_data, outputs_mc, options, key='npv', justIncl=False, show=True, save=True)
# plotEffComparison(outputs_data, outputs_mc, options, key='eta', justIncl=False, show=True, save=True)
# plotEffComparison(outputs_data, outputs_mc, options, key='diept', justIncl=False, show=True)


# ## Write SF JSON


import pandas as pd
from IPython.display import display

def saveEffScaleFactors(key='pt', save=False):
    output_dict = {}
    out_data_effs = []
    out_mc_effs = []
    data_acc = []
    mc_acc = []
    for output_data, output_mc in zip(outputs_data,outputs_mc):
        eff_output_data = output_data[key]
        eff_output_mc = output_mc[key]
        bins = options[key]['bins']
        xlabel = options[key]['xlabel']

        xs = (bins[1:] + bins[:-1]) / 2
        # x_err=np.diff(bins)/2
        
        title, nums_data, num_errs_data, denoms_data, denom_errs_data = np.array([[*i] for i in eff_output_data]).T
        title, nums_mc, num_errs_mc, denoms_mc, denom_errs_mc = np.array([[*i] for i in eff_output_mc]).T

        title = np.asarray(title)[np.nonzero(np.asarray(title))][0] if len(np.asarray(title)[np.nonzero(np.asarray(title))]) else ''
        # title = title.replace('Incl', 'Excl')

        nums_data = nums_data.astype(np.double)
        num_errs_data = num_errs_data.astype(np.double)
        denoms_data = denoms_data.astype(np.double)
        denom_errs_data = denom_errs_data.astype(np.double)

        nums_mc = nums_mc.astype(np.double)
        num_errs_mc = num_errs_mc.astype(np.double)
        denoms_mc = denoms_mc.astype(np.double)
        denom_errs_mc = denom_errs_mc.astype(np.double)

        data_acc.append(nums_data)
        mc_acc.append(nums_mc)
        out_data_effs.append((100*np.divide(nums_data, denoms_data, out=np.zeros_like(nums_data), where=denoms_data!=0)).round(1))
        out_mc_effs.append((100*np.divide(nums_mc, denoms_mc, out=np.zeros_like(nums_mc), where=denoms_mc!=0)).round(1))
        sfs = []
        for num_data, err_num_data, denom_data, err_denom_data, num_mc, err_num_mc, denom_mc, err_denom_mc in zip(nums_data, num_errs_data, denoms_data, denom_errs_data, nums_mc, num_errs_mc, denoms_mc, denom_errs_mc):
            eff_data = num_data / denom_data if denom_data != 0 else np.NaN
            eff_mc = num_mc / denom_mc if denom_mc != 0 else np.NaN
            # print(title, key, eff_data, eff_mc)
            if not np.isnan(eff_data) and not np.isnan(eff_mc):
                sf = eff_data / eff_mc if eff_mc else np.NaN
                # err_sf = sf * np.sqrt(((err_num_data / num_data)**2 * (err_denom_data / denom_data)**2 * (err_num_mc / num_mc)**2 * (err_denom_mc / denom_mc)**2))
                sfs.append((round(sf, 3), round(abs(1 - sf), 3)))
            else: sfs.append((np.NaN, np.NaN))
        output_dict[title] = {key : (bins[:-1]).tolist(), 'effs' : sfs}

    if save:
        with open('dietrig_scalefactors.json', 'w') as outfile:
            json.dump(output_dict, outfile, indent=4)

    out_data_df = pd.DataFrame( out_data_effs, 
                           columns=[f'{ibin} - {jbin}' for ibin, jbin in zip(bins[:-1],bins[1:])],
                           index=[list(output.values())[0][0][0] for output in outputs_data])
    out_mc_df = pd.DataFrame( out_mc_effs, 
                           columns=[f'{ibin} - {jbin}' for ibin, jbin in zip(bins[:-1],bins[1:])],
                           index=[list(output.values())[0][0][0] for output in outputs_data])

    return output_dict, out_data_df, out_mc_df, np.array(data_acc), np.array(mc_acc)

def add_wgt_row(df, wgts, name='Average'):
    avgs = np.array([lst for lst in [df[col].values[1:] for col in df]])
    int_avgs = [round(np.sum(i),1) for i in wgts * avgs]
    df = df.append(pd.DataFrame({key:avg for key,avg in zip(df.columns,int_avgs)}, index=[name], columns=df.columns))
    return df


# ## Write Eff JSONs

import pandas as pd
from IPython.display import display

def saveEffs(key='pt', save=False):
    output_dict = {}
    out_data_effs = []
    data_acc = []
    for output_data in outputs_data:
        eff_output_data = output_data[key]
        bins = options[key]['bins']
        xlabel = options[key]['xlabel']

        xs = (bins[1:] + bins[:-1]) / 2
        title, nums_data, num_errs_data, denoms_data, denom_errs_data = np.array([[*i] for i in eff_output_data]).T
        title = np.asarray(title)[np.nonzero(np.asarray(title))][0] if len(np.asarray(title)[np.nonzero(np.asarray(title))]) else ''
        # title = title.replace('Incl', 'Excl')

        nums_data = nums_data.astype(np.double)
        num_errs_data = num_errs_data.astype(np.double)
        denoms_data = denoms_data.astype(np.double)
        denom_errs_data = denom_errs_data.astype(np.double)

        data_acc.append(nums_data)
        out_data_effs.append((100*np.divide(nums_data, denoms_data, out=np.zeros_like(nums_data), where=denoms_data!=0)).round(1))
        effs = []
        for num_data, err_num_data, denom_data, err_denom_data in zip(nums_data, num_errs_data, denoms_data, denom_errs_data):
            eff_data = num_data / denom_data if denom_data != 0 else np.NaN
            err_data = eff_data * np.sqrt((err_num_data / num_data)**2 * (err_denom_data / denom_data)**2) if (denom_data != 0 and num_data != 0) else np.NaN
            if not np.isnan(eff_data):
                effs.append((round(eff_data,3),round(err_data,3)))
            else: effs.append(np.NaN)
        output_dict[title] = {key : (bins[:-1]).tolist(), 'effs' : effs}

    if save:
        with open(save, 'w') as outfile:
            json.dump(output_dict, outfile, indent=4)

    out_data_df = pd.DataFrame( out_data_effs, 
                           columns=[f'{ibin} - {jbin}' for ibin, jbin in zip(bins[:-1],bins[1:])],
                           index=[list(output.values())[0][0][0] for output in outputs_data])


    return output_dict, out_data_df, np.array(data_acc)

def saveEffsMC(key='pt', save=False):
    output_dict = {}
    out_mc_effs = []
    mc_acc = []
    for output_mc in outputs_mc:
        eff_output_mc = output_mc[key]
        bins = options[key]['bins']
        xlabel = options[key]['xlabel']
        xs = (bins[1:] + bins[:-1]) / 2
        title, nums_mc, num_errs_mc, denoms_mc, denom_errs_mc = np.array([[*i] for i in eff_output_mc]).T
        title = np.asarray(title)[np.nonzero(np.asarray(title))][0] if len(np.asarray(title)[np.nonzero(np.asarray(title))]) else ''
        # title = title.replace('Incl', 'Excl')

        nums_mc = nums_mc.astype(np.double)
        num_errs_mc = num_errs_mc.astype(np.double)
        denoms_mc = denoms_mc.astype(np.double)
        denom_errs_mc = denom_errs_mc.astype(np.double)

        mc_acc.append(nums_mc)
        out_mc_effs.append((100*np.divide(nums_mc, denoms_mc, out=np.zeros_like(nums_mc), where=denoms_mc!=0)).round(1))
        effs = []
        for num_mc, err_num_mc, denom_mc, err_denom_mc in zip(nums_mc, num_errs_mc, denoms_mc, denom_errs_mc):
            eff_mc = num_mc / denom_mc if denom_mc != 0 else np.NaN
            err_mc = eff_mc * np.sqrt((err_num_mc / num_mc)**2 * (err_denom_mc / denom_mc)**2) if (denom_mc != 0 and num_mc != 0) else np.NaN

            if not np.isnan(eff_mc):
                effs.append((round(eff_mc,3),round(err_mc,3)))
            else: effs.append(np.NaN)
        output_dict[title] = {key : (bins[:-1]).tolist(), 'effs' : effs}

    if save:
        with open(save, 'w') as outfile:
            json.dump(output_dict, outfile, indent=4)

    out_mc_df = pd.DataFrame(out_mc_effs, 
                           columns=[f'{ibin} - {jbin}' for ibin, jbin in zip(bins[:-1],bins[1:])],
                           index=[list(output.values())[0][0][0] for output in outputs_mc])


    return output_dict, out_mc_df, np.array(mc_acc)

def add_wgt_row(df, wgts, name='Average'):
    avgs = np.array([lst for lst in [df[col].values[1:] for col in df]])
    int_avgs = [round(np.sum(i),1) for i in wgts * avgs]
    df = df.append(pd.DataFrame({key:avg for key,avg in zip(df.columns,int_avgs)}, index=[name], columns=df.columns))
    return df


# In[34]:


sf_dict, sf_data_df, sf_mc_df, sf_data_acc, sf_mc_acc = saveEffScaleFactors(key='eta',save=True)
data_eff_dict, data_df, data_acc = saveEffs(key='eta',save='Trigger_Effs/TrigEffs_Incl_5_11_23_etabinned.json')
mc_eff_dict, mc_df, mc_acc = saveEffsMC(key='eta',save='Trigger_Effs/TrigEffs_MC_Incl_5_11_23_etabinned.json')

# data_acc_wgts = np.array([a / np.sum(a) for a in data_acc[1:].T])
# incl_lumi_wgts = np.array([ 1.5349e-01, 1.4637e-01, 1.4121e-01, 1.4075e-01, 1.0062e-01,
#                        8.5480e-02, 8.2440e-02, 5.1190e-02, 4.3760e-02, 3.1700e-02,
#                        1.5330e-02, 3.9500e-03, 3.2600e-03, 3.2000e-04, 1.4000e-04 ])

# # df_data = add_wgt_row(df_data, incl_lumi_wgts, 'Lumi Weighted Avg')
# df_data = add_wgt_row(df_data, data_acc_wgts, 'Acc Weighted Avg')
# mc_acc_wgts = np.array([a / np.sum(a) for a in mc_acc[1:].T])
# df_mc = add_wgt_row(df_mc, mc_acc_wgts, 'Acc Weighted Avg')


# In[ ]:


# histOut_binned_L1_11_HLT_6p5_Skim.root
# histOut_binned_L1_10p5_HLT_6p5_Skim.root
# histOut_binned_L1_10p5_HLT_5_Skim.root
# histOut_binned_L1_8p5_HLT_5_Skim.root
# histOut_binned_L1_8_HLT_5_Skim.root
# histOut_binned_L1_7_HLT_5_Skim.root
# histOut_binned_L1_6p5_HLT_4p5_Skim.root
# histOut_binned_L1_6_HLT_4_Skim.root
# histOut_binned_L1_5p5_HLT_6_Skim.root
# histOut_binned_L1_5p5_HLT_4_Skim.root
# histOut_binned_Incl_Skim.root

f1 = ROOT.TFile('histOut_binned_Incl_Skim_pt7p4to10p5.root')
f2 = ROOT.TFile('histOut_mc_pt7p4to10p5.root')
plots = ['sublead_el_pt', 'sublead_el_eta', 'sublead_el_phi', 'diel_m']
ROOT.gStyle.SetOptStat(0)

h_1 = f1.plots.Get('lead_el_pt')
# h_2 = f1.plots.Get('sublead_el_pt')
h_3 = f2.plots.Get('lead_el_pt')
# h_4 = f2.plots.Get('sublead_el_pt')

h_1.Rebin(4)
# h_2.Rebin(4)
h_3.Rebin(4)
# h_4.Rebin(4)

h_3.GetXaxis().SetRangeUser(0,50)
# h_4.SetMaximum(.25)
h_1.Scale(1/f1.plots.lead_el_pt.GetEntries())
# h_2.Scale(1/f1.plots.sublead_el_pt.GetEntries())
h_3.Scale(1/f2.plots.lead_el_pt.GetEntries())
# h_4.Scale(1/f2.plots.sublead_el_pt.GetEntries())

# print(h_1.GetSumOfWeights())
# print(h_2.GetSumOfWeights())
# print(h_3.GetSumOfWeights())
# print(h_4.GetSumOfWeights())

# h_4.SetTitle('')
c = ROOT.TCanvas('c', 'c', 600,600)
# leg = ROOT.TLegend(.4,.12,.6,.3)
leg = ROOT.TLegend(.65,.65,.88,.88)
# ROOT.gPad.SetLogy()
h_1.SetLineColor(600)
h_1.SetLineWidth(2)
# h_2.SetLineColor(600)
# h_2.SetLineWidth(2)
# h_2.SetLineStyle(2)
h_3.SetLineColor(632)
h_3.SetLineWidth(2)
# h_4.SetLineColor(632)
# h_4.SetLineWidth(2)
# h_4.SetLineStyle(2)

leg.AddEntry(h_1, '[Data]', 'le')
# leg.AddEntry(h_2, 'Sublead p_{T} [Data]', 'le')
leg.AddEntry(h_3, '[MC]', 'le')
# leg.AddEntry(h_4, 'Sublead p_{T} [MC]', 'le')

# h_4.Draw('HIST E ')
# h_2.Draw('HIST E SAME')
h_3.Draw('HIST E ')
h_1.Draw('HIST E SAME')
leg.Draw()
ROOT.gPad.Update()
c.Draw()


# In[ ]:


print('MC Bad Fits:')
print_badfits(badfits_mc)
print('Data Bad Fits:')
print_badfits(badfits_data)


# In[208]:


with open('JSON/Eras_CDEFG_Simplified/L1_6p5_HLT_4p5_Excl_Final.json','r') as f:
    path1_excl = json.load(f)
with open('JSON/Eras_CDEFG_Simplified/L1_8p0_HLT_5p0_Excl_Final.json','r') as f:
    path2_excl = json.load(f)
with open('JSON/Eras_CDEFG_Simplified/L1_9p0_HLT_6p0_Excl_Final.json','r') as f:
    path3_excl = json.load(f)
with open('JSON/Eras_CDEFG_Simplified/L1_6p5_HLT_4p5_Incl_Final.json','r') as f:
    path1_incl = json.load(f)
with open('JSON/Eras_CDEFG_Simplified/L1_8p0_HLT_5p0_Incl_Final.json','r') as f:
    path2_incl = json.load(f)
with open('JSON/Eras_CDEFG_Simplified/L1_9p0_HLT_6p0_Incl_Final.json','r') as f:
    path3_incl = json.load(f)

def mergeIntervals(arr1, arr2):
    arr = arr1
    arr.extend(arr2)

    arr.sort(key=lambda x: x[0])
    index = 0
 
    for i in range(1, len(arr)):
        if (arr[index][1] >= arr[i][0]):
            arr[index][1] = max(arr[index][1], arr[i][1])
        else:
            index = index + 1
            arr[index] = arr[i]
 
    out = []
    for i in range(index+1):
        out.append(arr[i])

    return out
    
def merge(A, B, f):
    # Start with symmetric difference; keys either in A or B, but not both
    merged = {k: A.get(k, B.get(k)) for k in A.keys() ^ B.keys()}
    # Update with `f()` applied to the intersection
    merged.update({k: f(A[k], B[k]) for k in A.keys() & B.keys()})
    return merged

excl_12 = merge(path1_excl, path2_excl, mergeIntervals)
excl_final = merge(excl_12, path3_excl, mergeIntervals)
with open('JSON/Eras_CDEFG_Simplified/trigger_OR_Excl.json', 'w') as write_file:
    json.dump(excl_final, write_file)

incl_12 = merge(path1_incl, path2_incl, mergeIntervals)
incl_final = merge(incl_12, path3_incl, mergeIntervals)
with open('JSON/Eras_CDEFG_Simplified/trigger_OR_Incl.json', 'w') as write_file:
    json.dump(incl_final, write_file)


# In[2]:


import ROOT
from root_plotting import *


# In[13]:


f = ROOT.TFile('../Data_MC_Comparisons/RootFiles/Data_triggerPU.root')
Hist = MultiHistPlot( init_params={
            'canvas_size'  : (1000,500),
            # 'title_string' : 'H_{T};;',
            'text_size'    : 'med',
            'y_title'      : 'nEvents [A.U.]',
            'xrange'       : (0.0, 70),
            # 'yrange'       : (0, 1),
            'rrange'       : (0, 10),
            'legtext_size' : 'med',
            'leg_pos'      : (.7,.3,.94,.89),
            'leg_scale'    : .2,
            'norm'         : 1,
            'colors'       : [900,890,880,870,860,850,840,830,820,810,800],
            'marker_style' : ['.','+','x','o','*','^','star'],
            'marker_size' : 'med',
})

plot = Hist.plotHists( 
    [
        # f.h_npv_L1_4p5_HLT_4p0,
        # f.h_npv_L1_5p0_HLT_4p0,
        # f.h_npv_L1_5p5_HLT_4p0,
        # f.h_npv_L1_5p5_HLT_6p0,
        f.h_npv_L1_6p0_HLT_4p0,
        f.h_npv_L1_6p5_HLT_4p5,
        f.h_npv_L1_7p0_HLT_5p0,
        f.h_npv_L1_7p5_HLT_5p0,
        f.h_npv_L1_8p0_HLT_5p0,
        # f.h_npv_L1_8p5_HLT_5p0,
        f.h_npv_L1_8p5_HLT_5p5,
        f.h_npv_L1_9p0_HLT_6p0,
        # f.h_npv_L1_10p5_HLT_5p0,
        f.h_npv_L1_10p5_HLT_6p5,
        f.h_npv_L1_11p0_HLT_6p5,
    ],
    titles = [
        # 'L1_4p5_HLT_4p0',
        # 'L1_5p0_HLT_4p0',
        # 'L1_5p5_HLT_4p0',
        # 'L1_5p5_HLT_6p0',
        'L1_6p0_HLT_4p0',
        'L1_6p5_HLT_4p5',
        'L1_7p0_HLT_5p0',
        'L1_7p5_HLT_5p0',
        'L1_8p0_HLT_5p0',
        # 'L1_8p5_HLT_5p0',
        'L1_8p5_HLT_5p5',
        'L1_9p0_HLT_6p0',
        # 'L1_10p5_HLT_5p0',
        'L1_10p5_HLT_6p5',
        'L1_11p0_HLT_6p5',
    ],
    ratio=True,
    show=True,
    )


# In[26]:


f0 = ROOT.TFile('../Data_MC_Comparisons/RootFiles/Rare_triggerPU.root')
f = ROOT.TFile('../Data_MC_Comparisons/RootFiles/Data_triggerPU.root')
Hist = MultiHistPlot( init_params={
            'canvas_size'  : (1000,500),
            # 'title_string' : 'H_{T};;',
            'text_size'    : 'med',
            'y_title'      : 'nEvents [A.U.]',
            'xrange'       : (0.0, 70),
            # 'yrange'       : (0, 1),
            'rrange'       : (.5, 5),
            'legtext_size' : 'med',
            'leg_pos'      : (.7,.3,.94,.89),
            'leg_scale'    : .2,
            # 'norm'         : 1,
            'colors'       : [900,890,880,870,860,850,840,830,820,810,800],
            'marker_style' : ['.','+','x','o','*','^','star'],
            'marker_size' : 'med',
})

plot = Hist.plotHists( 
    [
        # f0.h_npv_L1_9p0_HLT_6p0,
        f.h_npv_L1_4p5_HLT_4p0,
        f.h_npv_L1_5p0_HLT_4p0,
        f.h_npv_L1_5p5_HLT_4p0,
        f.h_npv_L1_5p5_HLT_6p0,
        f.h_npv_L1_6p0_HLT_4p0,
        f.h_npv_L1_6p5_HLT_4p5,
        f.h_npv_L1_7p0_HLT_5p0,
        f.h_npv_L1_7p5_HLT_5p0,
        f.h_npv_L1_8p0_HLT_5p0,
        f.h_npv_L1_8p5_HLT_5p0,
        f.h_npv_L1_8p5_HLT_5p5,
        f.h_npv_L1_9p0_HLT_6p0,
        f.h_npv_L1_10p5_HLT_5p0,
        f.h_npv_L1_10p5_HLT_6p5,
        f.h_npv_L1_11p0_HLT_6p5,
    ],
    titles = [
        # 'MC',
        'L1_4p5_HLT_4p0',
        'L1_5p0_HLT_4p0',
        'L1_5p5_HLT_4p0',
        'L1_5p5_HLT_6p0',
        'L1_6p0_HLT_4p0',
        'L1_6p5_HLT_4p5',
        'L1_7p0_HLT_5p0',
        'L1_7p5_HLT_5p0',
        'L1_8p0_HLT_5p0',
        'L1_8p5_HLT_5p0',
        'L1_8p5_HLT_5p5',
        'L1_9p0_HLT_6p0',
        'L1_10p5_HLT_5p0',
        'L1_10p5_HLT_6p5',
        'L1_11p0_HLT_6p5',
    ],
    ratio=False,
    show=True,
    )


# In[19]:


f = ROOT.TFile('../Data_MC_Comparisons/RootFiles/jpsi_mass.root')
# Hist = HistPlot( init_params={
#             'canvas_size'  : (1000,500),
#             # 'title_string' : 'H_{T};;',
#             'text_size'    : 'med',
#             'y_title'      : 'nEvents [A.U.]',
#             'xrange'       : (0.0, 70),
#             # 'yrange'       : (0, 1),
#             'rrange'       : (0, 10),
#             'legtext_size' : 'med',
#             'leg_pos'      : (.7,.3,.94,.89),
#             'leg_scale'    : .2,
#             'norm'         : 1,
#             'colors'       : [900,890,880,870,860,850,840,830,820,810,800],
#             'marker_style' : ['.','+','x','o','*','^','star'],
#             'marker_size' : 'med',
# })
# plot = Hist.plotHists(f.h_massll,show=True)

c = ROOT.TCanvas()
f.h_massll.Draw()
c.Draw()

