import os
import json
import math
import re
import yaml
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from uncertainties import ufloat
import argparse
import copy

import ROOT

from plotting_scripts.EfficiencyPlot import EfficiencyPlot


def set_verbosity(verb):
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kInfo if verb else ROOT.kWarning
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.INFO if verb else ROOT.RooFit.ERROR)
    printlevel = ROOT.RooFit.PrintLevel(1 if verb else -1)

    return printlevel


def get_event_count(h, xrange=None):
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

def assign_hist_format(h_name):
    if 'ptbinned' in h_name:
        h_format = {
            'bins' : np.array([5, 7, 9, 10, 11, 12, 13, 20], dtype=np.double),
            'xlabel' : 'Sublead Electron p_{T} [GeV]',
        }
    elif 'drbinned' in h_name:
        h_format = {
            'bins' : np.array([0, 0.12, 0.2, 0.28, 0.44, 1.], dtype=np.double),
            'xlabel' : r'\Delta R(e_{1},e_{2})',
        }
    elif 'etabinned' in h_name:
        h_format = {
            'bins' : np.array([-1.22, -0.7, -.2, 0.2, .7, 1.22], dtype=np.double),
            'xlabel' : r'Sublead \ Electron \ \eta',
        }
    else:
        raise ValueError(f'No formatting options for hist name {h_name}')

    return h_format


def do_fit(h, signal_only=False, savename=None, get_params=False, signal_params=None, bkg_poly_deg=2, printlevel=ROOT.RooFit.PrintLevel(-1)):

    data_yield = h.GetEntries()
    data_min = h.GetXaxis().GetXmin()
    data_max = h.GetXaxis().GetXmax()

    mass = ROOT.RooRealVar('mass', 'mass', 2.5, 3.5)
    data = ROOT.RooDataHist('data', 'data', mass, h)
    
    sig_coeff =  ROOT.RooRealVar('sig_coeff', 'sig_coeff', data_yield*.05, 0, data_yield)
    bkg_coeff =  ROOT.RooRealVar('bkg_coeff', 'bkg_coeff', data_yield*.05, 0, data_yield)
    if signal_params:
        dcb_mean =   ROOT.RooRealVar('dcb_mean', 'dcb_mean', signal_params['dcb_mean'], 0.95*signal_params['dcb_mean'], 1.05*signal_params['dcb_mean'])
        dcb_sigma =  ROOT.RooRealVar('dcb_sigma', 'dcb_sigma', signal_params['dcb_sigma'], 0.95*signal_params['dcb_sigma'], 1.05*signal_params['dcb_sigma'])
        dcb_alpha1 = ROOT.RooRealVar('dcb_alpha1', 'dcb_alpha1', signal_params['dcb_alpha1'], 0.95*signal_params['dcb_alpha1'], 1.05*signal_params['dcb_alpha1'])
        dcb_n1 =     ROOT.RooRealVar('dcb_n1', 'dcb_n1', signal_params['dcb_n1'], 0.95*signal_params['dcb_n1'], 1.05*signal_params['dcb_n1'])
        dcb_alpha2 = ROOT.RooRealVar('dcb_alpha2', 'dcb_alpha2', signal_params['dcb_alpha2'], 0.95*signal_params['dcb_alpha2'], 1.05*signal_params['dcb_alpha2'])
        dcb_n2 =     ROOT.RooRealVar('dcb_n2', 'dcb_n2', signal_params['dcb_n2'], 0.95*signal_params['dcb_n2'], 1.05*signal_params['dcb_n2'])
    else:
        dcb_mean =   ROOT.RooRealVar('dcb_mean', 'dcb_mean', 3.0969, 3.0, 3.2)
        dcb_sigma =  ROOT.RooRealVar('dcb_sigma', 'dcb_sigma', 0.046, 0.001, 10.)
        dcb_alpha1 = ROOT.RooRealVar('dcb_alpha1', 'dcb_alpha1', 1, 0.001, 5.)
        dcb_n1 =     ROOT.RooRealVar('dcb_n1', 'dcb_n1', 2., .01, 10.)
        dcb_alpha2 = ROOT.RooRealVar('dcb_alpha2', 'dcb_alpha2', 1, 0.001, 5.)
        dcb_n2 =     ROOT.RooRealVar('dcb_n2', 'dcb_n2', 2., .01, 10.)

    poly_pars = [ROOT.RooRealVar(f'poly_a{i}', f'poly_a{i}', *[0, -50,50]) for i in range(bkg_poly_deg)]
    poly_offset = ROOT.RooRealVar('poly_offset','poly_offset', 3, -10, 10)
    poly_diff = ROOT.RooFormulaVar('diff','mass-poly_offset', ROOT.RooArgList(mass, poly_offset))

    sig_pdf = ROOT.RooCrystalBall('sig_pdf', 'Signal Fit', mass, dcb_mean, dcb_sigma, dcb_alpha1, dcb_n1, dcb_alpha2, dcb_n2)
    bkg_pdf = ROOT.RooPolynomial('bkg_pdf', 'Background Fit', poly_diff, ROOT.RooArgList(*poly_pars))
    
    if signal_only: 
        fit_model = ROOT.RooAddPdf('fit_model', 'Signal Fit', ROOT.RooArgList(sig_pdf), ROOT.RooArgList(sig_coeff))
    else: 
        fit_model = ROOT.RooAddPdf('fit_model', 'Signal + Background Fit', ROOT.RooArgList(sig_pdf, bkg_pdf), ROOT.RooArgList(sig_coeff, bkg_coeff))

    fit_result = fit_model.fitTo(data, ROOT.RooFit.Save(True), ROOT.RooFit.SumW2Error(True), printlevel)

    out = [sig_coeff.getVal(), sig_coeff.getError()]
    params_output = {
        'dcb_mean'   : dcb_mean.getVal(),
        'dcb_sigma'  : dcb_sigma.getVal(),
        'dcb_alpha1' : dcb_alpha1.getVal(),
        'dcb_n1'     : dcb_n1.getVal(),
        'dcb_alpha2' : dcb_alpha2.getVal(),
        'dcb_n2'     : dcb_n2.getVal(),
    }

    if not signal_only: 
        out.extend([bkg_coeff.getVal(), bkg_coeff.getError()])
        params_output.update({
            'poly_offset'  : poly_offset.getVal(),
            **{i.GetName() : i for i in poly_pars},
        })
    
    out += [params_output] if get_params else []

    frame = mass.frame(ROOT.RooFit.Title(' '))
    data.plotOn(frame)
    
    if signal_only:
        fit_model.plotOn(frame, ROOT.RooFit.Name(fit_model.GetName()), ROOT.RooFit.LineColor(38), ROOT.RooFit.Normalization(sig_coeff.getVal(),ROOT.RooAbsReal.NumEvent))
        h_pull = frame.pullHist()

    else:
        fit_model.plotOn(frame, ROOT.RooFit.Name(fit_model.GetName()), ROOT.RooFit.LineColor(38), ROOT.RooFit.Normalization(sig_coeff.getVal()+bkg_coeff.getVal(),ROOT.RooAbsReal.NumEvent))
        h_pull = frame.pullHist()
        fit_model.plotOn(frame, ROOT.RooFit.Components('sig_pdf'), ROOT.RooFit.LineColor(32), ROOT.RooFit.LineStyle(ROOT.kDashed))
        fit_model.plotOn(frame, ROOT.RooFit.Components('bkg_pdf'), ROOT.RooFit.LineColor(46), ROOT.RooFit.LineStyle(ROOT.kDashed))

    # fit_model.paramOn(frame, ROOT.RooFit.Layout(0.12, 0.3, 0.88), ROOT.RooFit.Format('NE', ROOT.RooFit.FixedPrecision(3)))

    frame_pull = mass.frame(ROOT.RooFit.Title(' '))
    frame_pull.addPlotable(h_pull, 'P')

    ndf = frame.GetXaxis().GetNbins() - 2 - len(fit_result.floatParsFinal())
    chi2 = frame.chiSquare('fit_model', 'h_data', len(fit_result.floatParsFinal()))
    chi2_text = ROOT.TLatex(0.7, 0.8, '#chi^{{2}}/ndf = {}'.format(round(chi2,1)))
    chi2_text.SetTextSize(0.05)
    chi2_text.SetNDC(ROOT.kTRUE)

    text = ROOT.TLatex(0.7, 0.7, f'N_{{J/#psi}} = {round(sig_coeff.getVal())} #pm {round(sig_coeff.getError(),1)}')
    text.SetTextSize(0.05)
    text.SetNDC(ROOT.kTRUE)

    c = ROOT.TCanvas('c', ' ', 800, 600)
    pad1 = ROOT.TPad('pad1', 'pad1', 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0.02)
    pad1.SetGridx()
    pad1.Draw()
    c.cd()
    pad2 = ROOT.TPad('pad2', 'pad2', 0, 0.05, 1, 0.3)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.2)
    pad2.SetGridx()
    pad2.Draw()

    pad1.cd()
    frame.Draw()
    ax_y_main = frame.GetYaxis()
    ax_x_main = frame.GetXaxis()
    ax_x_main.SetLabelOffset(3.)

    chi2_text.Draw()
    text.Draw()

    pad2.cd()
    frame_pull.Draw()

    ax_y_pull = frame_pull.GetYaxis()
    ax_x_pull = frame_pull.GetXaxis()

    line = ROOT.TLine(ax_x_pull.GetXmin(), 0, ax_x_pull.GetXmax(), 0)
    line.SetLineStyle(7)
    line.Draw()

    ax_y_pull.SetTitle('#frac{y - y_{fit}}{#sigma_{y}}')
    ax_y_pull.SetTitleOffset(.35)
    ax_y_pull.SetNdivisions(8)

    ax_y_pull.SetTitleSize(2.8*ax_y_main.GetTitleSize())
    ax_y_pull.SetLabelSize(2.8*ax_y_main.GetLabelSize())
    ax_x_pull.SetTitleSize(2.8*ax_x_main.GetTitleSize())
    ax_x_pull.SetLabelSize(2.8*ax_x_main.GetLabelSize())


    c.SaveAs(str(savename))
    c.Close()

    return out


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
        if key in self: del self[key]
        else: raise AttributeError(f"No attribute named {key}")


def make_plotlist(cfg):
    plots = []
    out_path = Path(cfg.output.output_dir) / 'test' if cfg.test else Path(cfg.output.output_dir)
    data_in_path = Path(cfg.inputs.data_dir).parent / 'test' if cfg.test else Path(cfg.inputs.data_dir)
    mc_in_path = Path(cfg.inputs.mc_dir).parent / 'test' if cfg.test else Path(cfg.inputs.mc_dir)
    for plot_name, plot_dict in cfg.plots.items():
        if cfg.test and ('test' not in plot_name):
            continue
        for trigger in plot_dict.triggers:
            for var in plot_dict.variables:
                plots.append(DotDict({
                    'name'        : '_'.join([plot_name,trigger,f'{var}binned']),
                    'trigger'     : trigger,
                    'data_file'   : data_in_path / plot_dict.files.data,
                    'mc_file'     : mc_in_path / plot_dict.files.mc,
                    'num_hist'    : cfg.inputs.hist_dir.strip('/')+'_'.join(['/diel_m',trigger,'num',f'{var}binned']),
                    'denom_hist'  : cfg.inputs.hist_dir.strip('/')+'_'.join(['/diel_m',trigger,'denom',f'{var}binned']),
                    'output_file' : out_path / Path('_'.join(['eff',trigger,f'{var}binned'])).with_suffix('.pdf'),
                }))
    
    return plots


def get_hists(cfg):
    hists = []
    hist_paths = [(cfg.data_file, (cfg.num_hist,cfg.denom_hist)),(cfg.mc_file, (cfg.num_hist,cfg.denom_hist))]
    for (fname, (num_path, denom_path)) in hist_paths:
        subhists = []
        f = ROOT.TFile(str(fname))
        num_h = f.Get(num_path)
        denom_h = f.Get(denom_path)
        nbins = num_h.GetYaxis().GetNbins()
        for i in range(1, nbins + 1):
            try: num_bin = num_h.ProjectionX('num_bin',i,i+1)
            except AttributeError('Cannot Find Numerator Histogram'): continue
            try: denom_bin = denom_h.ProjectionX('denom_bin',i,i+1)
            except AttributeError('Cannot Find Denominator Histogram'): continue
            subhists.append((copy.deepcopy(num_bin),copy.deepcopy(denom_bin)))
        hists.append(subhists)
    return hists

def make_eff_plot_dict(cfg):
    plot_list = make_plotlist(cfg)

    eff_plot_list = []
    for plot_cfg in plot_list:
        if cfg.test and ('test' not in plot_cfg.name):
            continue
        elif not cfg.test and ('test' in plot_cfg.name):
            continue

        print(f'processing hists for {plot_cfg.name}')

        eff_dict = {
            'name' : plot_cfg.name,
            'trigger' : plot_cfg.trigger,
            'output_file' : plot_cfg.output_file,
            'data_num_yields' : [],
            'data_denom_yields' : [],
            'mc_num_yields' : [],
            'mc_denom_yields' : [],
            **assign_hist_format(plot_cfg.name),
        }

        data_hists, mc_hists = get_hists(plot_cfg)
        fit_output_file = (plot_cfg.output_file.parent / 'fits') / plot_cfg.output_file.name
        for i, ((data_num_hist, data_denom_hist), (mc_num_hist, mc_denom_hist)) in enumerate(zip(data_hists, mc_hists)):
            n_num_mc, n_num_mc_err, sig_params = do_fit(
                mc_num_hist, 
                signal_only=True, 
                savename=fit_output_file.with_stem(f'mc_num_fit_{plot_cfg.name}_bin{i}'), 
                get_params=True,
                printlevel=cfg.printlevel,
            )

            n_num_data, n_num_data_err, _, _ = do_fit(
                data_num_hist, 
                signal_params=sig_params,
                savename=fit_output_file.with_stem(f'data_num_fit_{plot_cfg.name}_bin{i}'), 
                printlevel=cfg.printlevel,
            )
        
            n_denom_mc, n_denom_mc_err, sig_params = do_fit(
                mc_denom_hist, 
                signal_only=True, 
                savename=fit_output_file.with_stem(f'mc_denom_fit_{plot_cfg.name}_bin{i}'), 
                get_params=True,
                printlevel=cfg.printlevel,
            )

            n_denom_data, n_denom_data_err, _, _ = do_fit(
                data_denom_hist, 
                signal_params=sig_params,
                savename=fit_output_file.with_stem(f'data_denom_fit_{plot_cfg.name}_bin{i}'), 
                printlevel=cfg.printlevel,
            )
            
            eff_dict['data_num_yields'].append((n_num_data, n_num_data_err))
            eff_dict['data_denom_yields'].append((n_denom_data, n_denom_data_err))
            eff_dict['mc_num_yields'].append((n_num_mc, n_num_mc_err))
            eff_dict['mc_denom_yields'].append((n_denom_mc, n_denom_mc_err))

        eff_plot_list.append(eff_dict)

    with open('eff_plot_data.pkl', 'wb') as f:
        pkl.dump(eff_plot_list, f)

    return eff_plot_list


def plot_efficiencies(eff_dicts, test=False):
    for d in eff_dicts:
        
        if test and ('test' not in d['name']):
            continue
        elif not test and ('test' in d['name']):
            continue

        bins = d['bins']
 
        h_num_data = ROOT.TH1F('h_num_data', 'h_num_data', len(bins)-1, bins)
        h_denom_data = ROOT.TH1F('h_denom_data', 'h_denom_data', len(bins)-1, bins)
        h_num_mc = ROOT.TH1F('h_num_mc', 'h_num_mc', len(bins)-1, bins)
        h_denom_mc = ROOT.TH1F('h_denom_mc', 'h_denom_mc', len(bins)-1, bins)

        zipped_yields = zip(
            d['data_num_yields'],
            d['data_denom_yields'],
            d['mc_num_yields'],
            d['mc_denom_yields']
        )
        for ibin, (data_num, data_denom, mc_num, mc_denom) in enumerate(zipped_yields):
            if (data_num > data_denom) or (mc_num > mc_denom): 
                continue

            h_num_data.SetBinContent(ibin+1, data_num[0])
            h_num_data.SetBinError(ibin+1, data_num[1])
            h_denom_data.SetBinContent(ibin+1, data_denom[0])
            h_denom_data.SetBinError(ibin+1, data_denom[1])
            h_num_mc.SetBinContent(ibin+1, mc_num[0])
            h_num_mc.SetBinError(ibin+1, mc_num[1])
            h_denom_mc.SetBinContent(ibin+1, mc_denom[0])
            h_denom_mc.SetBinError(ibin+1, mc_denom[1])

        
        eff_data = ROOT.TEfficiency(h_num_data, h_denom_data)
        eff_data.SetStatisticOption(ROOT.TEfficiency.kBBayesian)

        eff_mc = ROOT.TEfficiency(h_num_mc, h_denom_mc)
        eff_mc.SetStatisticOption(ROOT.TEfficiency.kBBayesian)

        eff_plot = EfficiencyPlot(init_params={
            'title_string' : f' ;{d["xlabel"]};Efficiency', 
            'xrange' : (bins[0],bins[-1]),
            'yrange' : (0.,1.1),
            'leg_scale' : .65,
            'leg_header' : d['trigger'],
            #'rrange' : (.7,1.3) if justIncl else ((0,2) if 'dr' in key else (.2,3)),
        })

        eff_plot.plotEfficiencies(
            eff_data,
            eff_mc,
            ratio=True, 
            h1_title='ParkingDoubleMuonLowMass 2022 Data',
            h2_title='B^{+} #rightarrow J/#psi K^{+} MC',
            save=str(d['output_file']),
            addIntegral=False
        )


def make_sf_json(eff_dicts):
    outputs = {}
    for eff_dict in eff_dicts:
        binvar_match = re.search(r'(?<=_)([a-zA-Z]*pt)(?=binned(?:_|$))', eff_dict['name'])
        binvar = binvar_match.group(1)

        data_num = [ufloat(val, unc) for val, unc in eff_dict['data_num_yields']]
        data_denom = [ufloat(val, unc) for val, unc in eff_dict['data_denom_yields']]
        mc_num = [ufloat(val, unc) for val, unc in eff_dict['mc_num_yields']]
        mc_denom = [ufloat(val, unc) for val, unc in eff_dict['mc_denom_yields']]

        data_ratio = [n / d if d.n != 0 else ufloat(0, 0) for n, d in zip(data_num, data_denom)]
        mc_ratio = [n / d if d.n != 0 else ufloat(0, 0) for n, d in zip(mc_num, mc_denom)]

        sfs = [d / m if m.n != 0 else ufloat(0, 0) for d, m in zip(data_ratio, mc_ratio)]
        sfs = [(round(r.n,3), round(r.s,3)) for r in sfs]

        output = {
            eff_dict['trigger'] : {
                binvar : list(eff_dict['bins'])[:-1],
                'sfs'  : sfs,
            }
        }

        outputs[binvar] = copy.deepcopy(output)

    for var, output_dict in outputs.items():
        with open(f'sf_jsons/trigger_sfs_{var}binned.json', 'w') as outfile:
            json.dump(output_dict, outfile, indent=4)

def main(cfg):
    cfg = DotDict(cfg)
    
    if cfg.file is None:
        eff_dicts = make_eff_plot_dict(cfg)
    elif Path(cfg.file).is_file():
        with open(cfg.file,'rb') as f:
            eff_dicts = pkl.load(f)
    else:
        raise ValueError(f'Efficiency dict file {str(cfg.file)} is not readable')

    plot_efficiencies(eff_dicts, test=cfg.test)
    make_sf_json(eff_dicts)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_plot_cfg.yml', help='plot configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    parser.add_argument('-f', '--file', dest='file', default=None, help='make plots from pkl file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))

    os.makedirs(Path(cfg.output.output_dir), exist_ok=True)
    cfg.test = args.test
    cfg.verbose = args.verbose
    cfg.printlevel = set_verbosity(args.verbose)
    cfg.file = args.file

    main(cfg)
