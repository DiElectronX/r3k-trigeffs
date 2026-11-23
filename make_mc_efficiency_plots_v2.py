import os
import json
import math
import re
import yaml
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from uncertainties import ufloat
import argparse
import copy

import ROOT

from plotting_scripts.EfficiencyPlot import EfficiencyPlot


def check_parameter_limits(savename, fit_result):
    if not fit_result:
        return 

    params_at_limit = []
    final_params = fit_result.floatParsFinal()

    for i in range(final_params.getSize()):
        param = final_params.at(i)
        name = param.GetName()
        value = param.getVal()
        min_val = param.getMin()
        max_val = param.getMax()

        # Use a small tolerance for floating point comparison
        tolerance = 1e-6

        if abs(value - min_val) < tolerance:
            params_at_limit.append(f"  - {name} (at MIN limit: {min_val})")
        
        if abs(value - max_val) < tolerance:
            params_at_limit.append(f"  - {name} (at MAX limit: {max_val})")

    if params_at_limit:
        print(f"--- Boundary Check for: {savename} ---")
        print("WARNING: The following parameters have final values at their limits:")
        for param_info in params_at_limit:
            print(param_info)
        print(50*'-')


def set_verbosity(verb):
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kInfo if verb else ROOT.kError
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.INFO if verb else ROOT.RooFit.ERROR)
    printlevel = ROOT.RooFit.PrintLevel(1 if verb else -1)

    return printlevel


def get_event_count(h, xrange=None):
    nEvents = 0.0
    err = 0.0
    
    if xrange is not None:
        bin_low = h.GetXaxis().FindBin(xrange[0])
        bin_high = h.GetXaxis().FindBin(xrange[1])
        
        error_ref = ROOT.Double(0.0)
        
        nEvents = h.IntegralAndError(bin_low, bin_high, error_ref)
        err = float(error_ref)
    
    else:
        nEvents = h.GetSumOfWeights() 
        err = np.sqrt(nEvents)
        
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
            'bins' : np.array([5, 7, 9, 10, 11, 12, 13, 999], dtype=np.double),
            # 'bins' : np.array([5, 8, 11, 20], dtype=np.double),
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


def do_fit(h, signal_only=False, savename=None, get_params=False, signal_params=None, bkg_shape_deg=4, printlevel=ROOT.RooFit.PrintLevel(-1), chi2_threshold=15.0, max_tries=20):

    #data_yield = h.GetEntries()
    #data_min = h.GetXaxis().GetXmin()
    #data_max = h.GetXaxis().GetXmax()

    #mass = ROOT.RooRealVar('mass', 'mass', 2.5, 3.4)
    #data = ROOT.RooDataHist('data', 'data', mass, h)

    #sig_coeff =  ROOT.RooRealVar('sig_coeff', 'sig_coeff', data_yield*.03, 0, data_yield)

    data_min = h.GetXaxis().GetXmin()
    data_max = h.GetXaxis().GetXmax()

    mass = ROOT.RooRealVar('mass', 'mass', 2.5, 3.4)
    data = ROOT.RooDataHist('data', 'data', mass, h)

    # Use the actual fitted histogram content scale, not GetEntries().
    lo_bin = h.GetXaxis().FindBin(2.5 + 1e-6)
    hi_bin = h.GetXaxis().FindBin(3.4 - 1e-6)
    data_yield = max(float(h.Integral(lo_bin, hi_bin)), 1.0)

    # Crude but more stable signal-yield seed from the J/psi window.
    peak_lo = h.GetXaxis().FindBin(2.95)
    peak_hi = h.GetXaxis().FindBin(3.20)
    peak_yield = max(float(h.Integral(peak_lo, peak_hi)), 1.0)

    sig_start = min(max(0.25 * peak_yield, 0.02 * data_yield), 0.8 * data_yield)
    bkg_start = max(data_yield - sig_start, 0.01 * data_yield)

    yield_max = 1.5 * data_yield

    sig_coeff = ROOT.RooRealVar('sig_coeff', 'sig_coeff', sig_start, 0, yield_max)

    cb_mean =   ROOT.RooRealVar('cb_mean', 'cb_mean', 3.0969)
    cb_sigma =  ROOT.RooRealVar('cb_sigma', 'cb_sigma', 0.046, 0.02, 0.1)
    cb_alpha = ROOT.RooRealVar('cb_alpha', 'cb_alpha', 2.0, 0.3, 10.)
    cb_n =     ROOT.RooRealVar('cb_n', 'cb_n', 8.0, 1.01, 40.)
    sig_pdf = ROOT.RooCrystalBall('sig_pdf', 'Signal Fit', mass, cb_mean, cb_sigma, cb_alpha, cb_n)

    #bkg_coeff =  ROOT.RooRealVar('bkg_coeff', 'bkg_coeff', data_yield*.97, 0, data_yield)
    bkg_coeff = ROOT.RooRealVar('bkg_coeff', 'bkg_coeff', bkg_start, 0, yield_max)
    bern_pars = [ROOT.RooRealVar(f'bern_p{i}', f'bern_p{i}', 1, 0, 10) for i in range(bkg_shape_deg)]
    bkg_pdf = ROOT.RooBernstein('bkg_pdf', 'Background Fit', mass, ROOT.RooArgList(*bern_pars))

    # Tail constraint
    n_central_val = ROOT.RooConstVar("n_central_val", "n_central_val", 10.0) # Target a large n
    n_sigma_val = ROOT.RooConstVar("n_sigma_val", "n_sigma_val", 3.0)   # With a reasonably strong pull
    n_constraint_pdf = ROOT.RooGaussian(
        "n_constraint_pdf", "Constraint on cb_n",
        cb_n, n_central_val, n_sigma_val
    )

    if signal_only:
        fit_model = ROOT.RooAddPdf('fit_model', 'Signal Fit', ROOT.RooArgList(sig_pdf), ROOT.RooArgList(sig_coeff))
    else:
        fit_model = ROOT.RooAddPdf('fit_model', 'Signal + Background Fit', ROOT.RooArgList(sig_pdf, bkg_pdf), ROOT.RooArgList(sig_coeff, bkg_coeff))


    fit_result = None
    chi2 = float('inf')
    attempt = 0
    sig_unc_ratio = 1
    good_fit = False

    while not good_fit and attempt < max_tries:
        if fit_result:
            fit_result.Delete()

        #fit_result = fit_model.fitTo(
        #    data,
        #    ROOT.RooFit.Save(True),
        #    ROOT.RooFit.Minos(True),
        #    ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(n_constraint_pdf)),
        #    printlevel
        #)

        fit_result = fit_model.fitTo(
            data,
            ROOT.RooFit.Save(True),
            ROOT.RooFit.Minos(False),
            ROOT.RooFit.Hesse(True),
            ROOT.RooFit.Strategy(2),
            ROOT.RooFit.ExternalConstraints(ROOT.RooArgSet(n_constraint_pdf)),
            printlevel
        )

        fit_status = fit_result.status()
        cov_qual = fit_result.covQual()
        status_ok = (fit_status == 0) or (fit_status == 1 and cov_qual >= 2)

        
        if fit_status != 0 or cov_qual < 2:
            print(
            f"[fit retry] {savename.stem}: "
            f"attempt={attempt}, "
            f"entries={h.GetEntries():.0f}, "
            f"sumw={h.GetSumOfWeights():.1f}, "
            f"status={fit_status}, "
            f"covQual={cov_qual}, "
            f"sig={sig_coeff.getVal():.2f}, "
            f"sigErr={sig_coeff.getError():.2f}"
        )

        if status_ok and cov_qual >= 2 and sig_coeff.getError() > 0:
        #if fit_status == 0 and cov_qual >= 2:
            sig_unc_ratio = sig_coeff.getError() / sig_coeff.getVal() if sig_coeff.getVal() > 0 else float('inf')
            
            tmp_frame = mass.frame(ROOT.RooFit.Title('tmp'))
            data.plotOn(tmp_frame)
            if signal_only:
                fit_model.plotOn(tmp_frame, ROOT.RooFit.Normalization(sig_coeff.getVal(),ROOT.RooAbsReal.NumEvent))
            else:
                fit_model.plotOn(tmp_frame, ROOT.RooFit.Normalization(sig_coeff.getVal()+bkg_coeff.getVal(),ROOT.RooAbsReal.NumEvent))
            
            chi2 = tmp_frame.chiSquare('fit_model_Norm[mass]', 'h_data', len(fit_result.floatParsFinal()))
            tmp_frame.Delete()

            if not math.isnan(chi2) and chi2 < chi2_threshold and sig_unc_ratio < 1:
                good_fit = True
            
        if fit_result and not good_fit:
            params_to_randomize = [cb_sigma, cb_alpha, cb_n] + bern_pars 
            for param in params_to_randomize:
                current_val = param.getVal()
                min_val = param.getMin()
                max_val = param.getMax()
                
                perturbation_range = (max_val - min_val) * 0.1
                random_shift = np.random.uniform(-perturbation_range, perturbation_range)
                new_val = current_val + random_shift
                new_val = max(min_val, min(max_val, new_val))
                
                param.setVal(new_val)
                param.setError(0.0)
        
        attempt += 1

    if not good_fit:
        print(f"WARNING: Fit failed to converge for {savename.stem} after {max_tries} attempts.")

    if attempt >= max_tries:
        print(f"WARNING: Fit failed to converge for {savename.stem} after {max_tries} attempts.")
        if get_params:
            return [None, None, None, None, None] if not signal_only else [None, None, None]
        return [None, None, None, None] if not signal_only else [None, None]


    check_parameter_limits(savename, fit_result)
    out = [sig_coeff.getVal(), sig_coeff.getError()]
    params_output = {
        'cb_mean'   : cb_mean.getVal(),
        'cb_sigma'  : cb_sigma.getVal(),
        'cb_alpha' : cb_alpha.getVal(),
        'cb_n'     : cb_n.getVal(),
    }

    if not signal_only:
        out.extend([bkg_coeff.getVal(), bkg_coeff.getError()])
        params_output.update({
            # **{i.GetName() : i for i in poly_pars},
            **{i.GetName() : i for i in bern_pars},
        })

    out += [params_output] if get_params else []

    frame = mass.frame(ROOT.RooFit.Title(str(savename.stem)))
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

    n_params = len(fit_result.floatParsFinal()) if fit_result else 0
    chi2 = frame.chiSquare('fit_model', 'h_data', n_params) if not signal_only else frame.chiSquare('fit_model', 'h_data', n_params)
    # chi2 = frame.chiSquare()
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

"""
def make_plotlist(cfg):
    plots = []
    out_path = Path('./eff_hists/test') if cfg.test else Path(cfg.output.output_dir)
    data_in_path = Path(cfg.inputs.data_dir).parent / 'test' if cfg.test else Path(cfg.inputs.data_dir)
    mc_in_path = Path(cfg.inputs.mc_dir).parent / 'test' if cfg.test else Path(cfg.inputs.mc_dir)
    for plot_name, plot_dict in cfg.plots.items():
        # For new format: triggers is a dict with data_trigger
        data_trigger = plot_dict.triggers['data_trigger'] if isinstance(plot_dict.triggers, dict) else plot_dict.triggers[0]
        for var in plot_dict.variables:
            plots.append(DotDict({
                'name': '_'.join([plot_name, f'{var}binned']),
                'data_trigger': data_trigger,
                'data_file': data_in_path / plot_dict.files.data,
                'mc_file': mc_in_path / plot_dict.files.mc if hasattr(plot_dict.files, 'mc') else None,
                'output_file': out_path / Path('_'.join(['eff', data_trigger, f'{var}binned'])).with_suffix('.pdf'),
                'mc_triggers': plot_dict.mc_triggers if hasattr(plot_dict, 'mc_triggers') else [],
                'var': var,
                **assign_hist_format(f'{var}binned'),
            }))

    return plots
"""

def make_plotlist(cfg):
    plots = []

    out_path = Path('./eff_hists/test') if cfg.test else Path(cfg.output.output_dir)
    data_in_path = Path(cfg.inputs.data_dir).parent / 'test' if cfg.test else Path(cfg.inputs.data_dir)
    mc_in_path = Path(cfg.inputs.mc_dir).parent / 'test' if cfg.test else Path(cfg.inputs.mc_dir)

    for plot_name, plot_dict in cfg.plots.items():

        # Support both old and new config formats
        if isinstance(plot_dict.triggers, dict):
            triggers_cfg = plot_dict.triggers
            data_trigger = triggers_cfg['data_trigger']
            mc_triggers = triggers_cfg.get('mc_triggers', plot_dict.get('mc_triggers', []))
        else:
            data_trigger = plot_dict.triggers[0]
            mc_triggers = plot_dict.get('mc_triggers', [])

        if not mc_triggers:
            raise RuntimeError(
                f"No mc_triggers found for plot '{plot_name}'. "
                "Check eff_plot_cfg_v2.yml nesting under triggers: mc_triggers."
            )

        weight_sum = sum(float(t['weight']) for t in mc_triggers)
        print(f"[trigeff] {plot_name}: {len(mc_triggers)} MC trigger paths, weight sum = {weight_sum:.6f}")

        # Fractions are rounded in the YAML, so do not use a 1e-6 tolerance.
        assert abs(weight_sum - 1.0) < 5e-1, (
            f"MC trigger weights for {plot_name} sum to {weight_sum}, not 1. "
            "Check eff_plot_cfg_v2.yml."
        )

        for var in plot_dict.variables:
            plots.append(DotDict({
                'name': '_'.join([plot_name, f'{var}binned']),
                'data_trigger': data_trigger,
                'data_file': data_in_path / plot_dict.files.data,
                'mc_file': mc_in_path / plot_dict.files.mc if hasattr(plot_dict.files, 'mc') else None,
                'output_file': out_path / Path('_'.join(['eff', data_trigger, f'{var}binned'])).with_suffix('.pdf'),
                'mc_triggers': mc_triggers,
                'var': var,
                **assign_hist_format(f'{var}binned'),
            }))

    return plots


def get_hists(cfg):
    # Only for data, since MC is handled per-path
    hists = []
    hist_paths = [(
        cfg.data_file, 
        f'hists/diel_m_{cfg.data_trigger}_num_{cfg.var}binned', 
        f'hists/diel_m_{cfg.data_trigger}_denom_{cfg.var}binned',
    )]
    for fname, num_path, denom_path in hist_paths:
        subhists = []
        f = ROOT.TFile(str(fname))
        num_h = f.Get(num_path)
        denom_h = f.Get(denom_path)
        nbins = num_h.GetYaxis().GetNbins()
        for i in range(1, nbins + 1):
            try:
                num_bin = num_h.ProjectionX('num_bin', i, i)
                # print(f'bin {i} num: ',num_bin.GetEntries())
            except AttributeError:
                continue
            try:
                denom_bin = denom_h.ProjectionX('denom_bin', i, i)
                # print(f'bin {i} deonom: ',denom_bin.GetEntries())
            except AttributeError:
                continue
            subhists.append((copy.deepcopy(num_bin), copy.deepcopy(denom_bin)))
        hists.append(subhists)
    return hists

def make_eff_plot_dict(cfg):
    plot_list = make_plotlist(cfg)
    eff_plot_list = []
    for plot_cfg in plot_list:
        print(f'processing hists for {plot_cfg.name}')
        eff_dict = {
            'name': plot_cfg.name,
            'data_trigger': plot_cfg.data_trigger,
            'output_file': plot_cfg.output_file,
            'data_num_yields': [],
            'data_denom_yields': [],
            'mc_eff_mixture': [],
            'mc_triggers': plot_cfg.mc_triggers,
            'bins': plot_cfg.bins,
            'xlabel': plot_cfg.xlabel,
        }
        # Data
        data_hists = get_hists(plot_cfg)
        fit_output_file = plot_cfg.output_file.parent / 'fits' / plot_cfg.output_file.name
        for i, (data_num_hist, data_denom_hist) in enumerate(data_hists[0]):
            num_results = do_fit(
                data_num_hist,
                savename=fit_output_file.with_stem(f'data_num_fit_{plot_cfg.name}_bin{i}'),
                printlevel=cfg.printlevel,
            )
            n_num_data, n_num_data_err = (num_results[0], num_results[1]) if num_results[0] is not None else (None, None)

            denom_results = do_fit(
                data_denom_hist,
                savename=fit_output_file.with_stem(f'data_denom_fit_{plot_cfg.name}_bin{i}'),
                printlevel=cfg.printlevel,
            )
            n_denom_data, n_denom_data_err = (denom_results[0], denom_results[1]) if denom_results[0] is not None else (None, None)

            eff_dict['data_num_yields'].append((n_num_data, n_num_data_err))
            eff_dict['data_denom_yields'].append((n_denom_data, n_denom_data_err))

        if plot_cfg.mc_file and plot_cfg.mc_triggers:
            mc_file = ROOT.TFile(str(plot_cfg.mc_file))
            n_bins = len(eff_dict['bins']) - 1
            mc_path_effs = []
            for mc_trig in plot_cfg.mc_triggers:
                path = mc_trig['path'].strip('"')
                weight = mc_trig['weight']
                num_hist_name = f'hists/diel_m_{path}_num_{plot_cfg.var}binned'
                denom_hist_name = f'hists/diel_m_{path}_denom_{plot_cfg.var}binned'
                num_hist = mc_file.Get(num_hist_name)
                denom_hist = mc_file.Get(denom_hist_name)
                effs = []
                for ibin in range(1, n_bins + 1):
                    #num_proj = num_hist.ProjectionX('num_bin', ibin, ibin)
                    #denom_proj = denom_hist.ProjectionX('denom_bin', ibin, ibin)
                    safe_path = path.replace('.', 'p').replace('-', '_')
                    num_proj = num_hist.ProjectionX(f'num_bin_mc_{safe_path}_{ibin}', ibin, ibin)
                    denom_proj = denom_hist.ProjectionX(f'denom_bin_mc_{safe_path}_{ibin}', ibin, ibin)
                    
                    if denom_proj.GetSumOfWeights() > 0:
                        nNum, nNum_err = get_event_count(num_proj)
                        nDenom, nDenom_err = get_event_count(denom_proj)
                        eff = ufloat(nNum, nNum_err) / ufloat(nDenom, nDenom_err)
                    else:
                        eff = ufloat(0, 0)
                    effs.append(eff * weight)
                mc_path_effs.append(effs)

            mc_eff_mixture = []
            for ibin in range(n_bins):
                eff_sum = sum(effs[ibin] for effs in mc_path_effs)
                mc_eff_mixture.append(eff_sum)
            eff_dict['mc_eff_mixture'] = mc_eff_mixture
        eff_plot_list.append(eff_dict)

    with open(Path(cfg.output.output_dir) / 'eff_plot_entries.pkl', 'wb') as f:
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

        # Fill data as before
        for ibin, (data_num, data_denom) in enumerate(zip(d['data_num_yields'], d['data_denom_yields'])):
            if data_num[0] is not None and data_denom[0] is not None:
                h_num_data.SetBinContent(ibin+1, data_num[0])
                h_num_data.SetBinError(ibin+1, data_num[1])
                h_denom_data.SetBinContent(ibin+1, data_denom[0])
                h_denom_data.SetBinError(ibin+1, data_denom[1])


        # MC: use mixture if present
        if 'mc_eff_mixture' in d and d['mc_eff_mixture']:
            for ibin, eff in enumerate(d['mc_eff_mixture']):
                # For TEfficiency, set numerator = eff * denom, error propagation is handled by ufloat
                denom = 1.0  # or set to 1 for all bins, since TEfficiency cares about ratio
                num = eff.n
                err = eff.s
                h_num_mc.SetBinContent(ibin+1, num)
                h_num_mc.SetBinError(ibin+1, err)
                h_denom_mc.SetBinContent(ibin+1, denom)
                h_denom_mc.SetBinError(ibin+1, 0)
        else:
            for ibin, (mc_num, mc_denom) in enumerate(zip(d['mc_num_yields'], d['mc_denom_yields'])):
                h_num_mc.SetBinContent(ibin+1, mc_num[0])
                h_num_mc.SetBinError(ibin+1, mc_num[1])
                h_denom_mc.SetBinContent(ibin+1, mc_denom[0])
                h_denom_mc.SetBinError(ibin+1, mc_denom[1])

        eff_data = ROOT.TEfficiency(h_num_data, h_denom_data)
        eff_data.SetStatisticOption(ROOT.TEfficiency.kBBayesian)
        eff_mc = ROOT.TEfficiency(h_num_mc, h_denom_mc)
        eff_mc.SetStatisticOption(ROOT.TEfficiency.kFNormal)

        # Make plot range nicer for visualization
        plot_xmax = 20 if bins[-1] >= 900 else bins[-1]

        eff_plot = EfficiencyPlot(init_params={
            'title_string': ' ;' + d["xlabel"] + ';Efficiency',
            #'xrange': (bins[0], bins[-1]),
            'xrange': (bins[0], plot_xmax),
            'yrange': (0., 1.1),
            'leg_scale': .65,
            'leg_header': '',
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


def make_sf_json(cfg, eff_dicts):
    outputs = {}
    for eff_dict in eff_dicts:
        matches = list(re.finditer(r'_([^_]+)binned(?=_|$)', eff_dict['name']))
        if not matches:
            raise ValueError(f"Could not find binvar in {eff_dict['name']}")

        binvar = matches[-1].group(1)
        # data_num = [ufloat(val, unc) for val, unc in eff_dict['data_num_yields']]
        # data_denom = [ufloat(val, unc) for val, unc in eff_dict['data_denom_yields']]
        # data_ratio = [n / d if d.n != 0 else ufloat(0, 0) for n, d in zip(data_num, data_denom)]
        # mc_ratio = eff_dict['mc_eff_mixture']
        # sfs = [d / m if m.n != 0 else ufloat(0, 0) for d, m in zip(data_ratio, mc_ratio)]

        # sfs = [(round(r.n,3), round(r.s,3)) for r in sfs]
        # data_ratio = [(round(r.n,3), round(r.s,3)) for r in data_ratio]
        # mc_ratio = [(round(r.n,3), round(r.s,3)) for r in mc_ratio]

        # output = {
        #     'binvar'    : binvar,
        #     'bins'      : list(eff_dict['bins'])[:-1],
        #     'sfs'       : sfs,
        #     'data_effs' : data_ratio,
        #     'mc_effs'   : mc_ratio,
        # }

        # outputs[eff_dict['name']] = outputs.get(eff_dict['name'], copy.deepcopy(output))

        data_num = [ufloat(val, unc) if val is not None else None for val, unc in eff_dict['data_num_yields']]
        data_denom = [ufloat(val, unc) if val is not None else None for val, unc in eff_dict['data_denom_yields']]

        # Propagate None through ratios
        data_ratio = [(n / d) if (n is not None and d is not None and d.n != 0) else None for n, d in zip(data_num, data_denom)]
        mc_ratio = eff_dict['mc_eff_mixture']
        sfs = [(d / m) if (d is not None and m.n != 0) else None for d, m in zip(data_ratio, mc_ratio)]

        # Format final output, preserving None
        sfs_out = [ (round(r.n, 3), round(r.s, 3)) if r is not None else None for r in sfs]
        data_ratio_out = [ (round(r.n, 3), round(r.s, 3)) if r is not None else None for r in data_ratio]
        mc_ratio_out = [ (round(r.n, 3), round(r.s, 3)) if r is not None else None for r in mc_ratio]

        edges = [float(x) for x in eff_dict['bins']]

        # Treat the high edge as inf for cuts.yml
        edges_for_cuts = edges[:]
        if edges_for_cuts[-1] >= 900:
            edges_for_cuts[-1] = "inf"

        cut_bins = []
        for i, sf in enumerate(sfs_out):
            lo = edges_for_cuts[i]
            hi = edges_for_cuts[i + 1]

            if sf is None:
                cut_bins.append([lo, hi, None, None])
            else:
                cut_bins.append([lo, hi, sf[0], sf[1]])

        """
        output = {
            'binvar'    : binvar,
            'bins'      : list(eff_dict['bins'])[:-1],
            'sfs'       : sfs_out,
            'data_effs' : data_ratio_out,
            'mc_effs'   : mc_ratio_out,
        }
        """

        output = {
            'binvar': binvar,
            'bin_edges': edges_for_cuts,
            'cut_bins': cut_bins,
            'sfs': sfs_out,
            'data_effs': data_ratio_out,
            'mc_effs': mc_ratio_out,
        }

        outputs[eff_dict['name']] = outputs.get(eff_dict['name'], copy.deepcopy(output))

    out_path = Path('eff_hists') / 'test' if cfg.test else Path(cfg.output.output_dir)
    with open(out_path / 'sf_jsons' / 'trigger_sfs.json', 'w') as outfile:
        json.dump(outputs, outfile, indent=4)

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
    make_sf_json(cfg, eff_dicts)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_plot_cfg_v2.yml', help='plot configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    parser.add_argument('-f', '--file', dest='file', default=None, help='make plots from pkl file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))

    os.makedirs(Path(cfg.output.output_dir), exist_ok=True)
    os.makedirs(Path(cfg.output.output_dir) / 'fits', exist_ok=True)
    os.makedirs(Path(cfg.output.output_dir) / 'sf_jsons', exist_ok=True)
    cfg.test = args.test
    cfg.verbose = args.verbose
    cfg.printlevel = set_verbosity(args.verbose)
    cfg.file = args.file

    main(cfg)

