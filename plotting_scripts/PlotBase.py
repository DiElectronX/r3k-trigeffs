import os
import numpy as np
from ROOT import gROOT, gStyle, gPad, TLegend, TEfficiency, TCanvas, TLine, TPad

gROOT.LoadMacro(os.path.join(os.getcwd(),'plotting_scripts/C_Files/tdrstyle.C'))
gROOT.LoadMacro(os.path.join(os.getcwd(),'plotting_scripts/C_Files/CMS_lumi.C'))

class PlotBase():
    def format_entry(self, hist, title=None, norm=None, marker_color='black', marker_style ='', marker_size='small', line_color='black', line_style='-', line_width='med'):
        if norm is not None:
            if hist.InheritsFrom(TEfficiency.Class()):
                raise TypeError('Cannot normalize TEfficiency')
            else:
                hist.Scale(norm/hist.GetEntries())
        
        color_map = {
            'white'   : 0,
            'black'   : 1,
            'red'     : 2,
            'green'   : 3,
            'blue'    : 4,
            'cyan'    : 432,
            'gray'    : 920,
            'magenta' : 616,
            'orange'  : 797,
        }

        linestyle_map = {
            '-'  : 1,
            '..' : 2,
            '--' : 9,
            '-.' : 10,
        }

        linewidth_map = {
            'thin'  : 1,
            'med'   : 2,
            'thick' : 5,
        }

        markerstyle_map = {
            ''     : 0,
            '.'    : 1,
            '+'    : 34,
            'x'    : 5,
            'o'    : 4,
            '*'    : 3,
            '^'    : 22,
            'star' : 29,
        }

        markersize_map = {
            'small'    : 1,
            'med'      : 1.5,
            'large'    : 3,
            'x-large'  : 4,
            'xx-large' : 5,
        }
        
        if title is not None: hist.SetTitle(title)
        else: gStyle.SetOptTitle(0)
        hist.SetMarkerColor(color_map[marker_color] if isinstance(marker_color, str) else marker_color)
        hist.SetMarkerStyle(markerstyle_map[marker_style])
        hist.SetMarkerSize(markersize_map[marker_size])
        hist.SetLineColor(color_map[line_color] if isinstance(line_color, str) else line_color)
        hist.SetLineStyle(linestyle_map[line_style])
        hist.SetLineWidth(linewidth_map[line_width])

    def format_axes(self, hist, option='full', xrange=None, yrange=None, text_size='small', title_string=None, x_title=None, y_title=None):
        titlesize_map = {
            'small' : .02,
            'med'   : .03,
            'large' : .05,
        }

        labelsize_map = {
            'small' : .025,
            'med'   : .03,
            'large' : .07,
        }

        titleoffset_map = {
            'hist' : {
                'small' : {'x' : 1.5, 'y' : 1.5},
                'med'   : {'x' : 1.6, 'y' : 1.8},
                'large' : {'x' : 1.5, 'y' : 1.5},
            },
            'eff' : {
                'small' : {'x' : 1.2, 'y' : 1.2},
                'med'   : {'x' : 1.2, 'y' : 1.1},
                'large' : {'x' : 1.2, 'y' : 1},
            },
            'ratio' : {
                'small' : {'x' : 1.5, 'y' : .5},
                'med'   : {'x' : 1.2, 'y' : .4},
                'large' : {'x' : 1.2, 'y' : .35},
            }
        }

        if option=='full':
            padh = padw = padsize = 1
            hist.Draw('E')
            gPad.Update()
            g = hist.GetPaintedGraph() if hist.InheritsFrom(TEfficiency.Class()) else hist

            # X- & Y-Axis
            if title_string is not None: g.SetTitle(title_string)
            if x_title is not None: g.GetXaxis().SetTitle(x_title)
            if y_title is not None: g.GetYaxis().SetTitle(y_title)
            if xrange is not None: g.GetXaxis().SetRangeUser(xrange[0],xrange[1])
            if yrange is not None:
                g.SetMinimum(yrange[0])
                g.SetMaximum(yrange[1])
            labelsize = labelsize_map[text_size] / padsize
            titlesize = titlesize_map[text_size] / padsize

            g.GetXaxis().SetLabelSize(labelsize)
            g.GetXaxis().SetLabelOffset(.008)
            g.GetXaxis().SetTitleSize(titlesize)
            g.GetXaxis().SetTitleOffset(titleoffset_map['hist'][text_size]['x'])
            g.GetYaxis().SetLabelSize(labelsize)
            g.GetYaxis().SetLabelOffset(.008)
            g.GetYaxis().SetTitleSize(titlesize)
            g.GetYaxis().SetTitleOffset(titleoffset_map['hist'][text_size]['y'])

        if option=='upper':
            padh = .7; padw = 1; padsize=.7
            # hist.Draw('E')
            gPad.Update()
            g = hist.GetPaintedGraph() if hist.InheritsFrom(TEfficiency.Class()) else hist
            labelsize = labelsize_map[text_size] / padsize
            titlesize = titlesize_map[text_size] / padsize

            # X-Axis
            if x_title is not None: g.GetXaxis().SetTitle(x_title)
            if xrange is not None: g.GetXaxis().SetRangeUser(xrange[0],xrange[1])
            g.GetXaxis().SetLabelSize(labelsize)
            g.GetXaxis().SetLabelOffset(.1)
            g.GetXaxis().SetTitleSize(titlesize)
            g.GetXaxis().SetTitleOffset(titleoffset_map['eff'][text_size]['x'])

            # Y-Axis
            if yrange is not None:
                g.SetMinimum(yrange[0])
                g.SetMaximum(yrange[1])
            g.GetYaxis().SetLabelSize(labelsize)
            g.GetYaxis().SetLabelOffset(.008)
            g.GetYaxis().SetTitleSize(titlesize)
            g.GetYaxis().SetTitleOffset(titleoffset_map['eff'][text_size]['y'])
            gPad.Update()

        elif option=='lower':
            padh = .25; padw = 1; padsize=.25
            hist.SetTitle('')
            labelsize = labelsize_map[text_size] / padsize
            titlesize = titlesize_map[text_size] / padsize
            
            # X-Axis
            if x_title is not None: g.GetXaxis().SetTitle(x_title)
            if xrange is not None: hist.GetXaxis().SetRangeUser(xrange[0],xrange[1])
            hist.GetXaxis().SetLabelSize(labelsize)
            hist.GetXaxis().SetLabelOffset(.008)
            hist.GetXaxis().SetTitleSize(titlesize)
            hist.GetXaxis().SetTitleOffset(titleoffset_map['ratio'][text_size]['x'])

            # Y-Axis
            hist.GetYaxis().SetTitle('Data / MC')
            if yrange is not None: hist.GetYaxis().SetRangeUser(yrange[0],yrange[1])
            hist.GetYaxis().SetLabelSize(labelsize)
            hist.GetYaxis().SetLabelOffset(.008)
            hist.GetYaxis().SetTitleSize(titlesize)
            hist.GetYaxis().SetTitleOffset(titleoffset_map['ratio'][text_size]['y'])
            hist.GetYaxis().SetNdivisions(4)

            # Draw Ratio Line at 1
            line = TLine(0,0,1,1)
            line.SetLineStyle(2)
            line.DrawLine(xrange[0],1,xrange[1],1) if xrange else line.DrawLine(0,1,hist.GetXaxis().GetXmax(),1)

    def format_legend(self, leg, pos='lower_right', option='full', scale=None, legtext_size=None):
        pos_map = {
            'full' : {
                'upper_left'   : [.16,.75,.79,.88],
                'upper_right'  : [.16,.75,.89,.88],
                'center_left'  : [.16,.43,.79,.56],
                'center_right' : [.16,.43,.89,.56],
                'lower_left'   : [.16,.17,.79,.3],
                'lower_right'  : [.16,.17,.89,.3],
            },
            'upper' : {
                'upper_left'   : [.16,.7,.79,.88],
                'upper_right'  : [.16,.65,.89,.88],
                'center_left'  : [.16,.4,.79,.55],
                'center_right' : [.16,.4,.89,.55],
                'lower_left'   : [.16,.03,.79,.2],
                'lower_right'  : [.16,.03,.89,.2],
            },
            'lower' : {
                'upper_left'   : [.16,.77,.5,.98],
                'upper_right'  : [.5,.77,.89,.98],
                'center_left'  : [.16,.65,.5,.85],
                'center_right' : [.5,.65,.89,.85],
                'lower_left'   : [.16,.52,.5,.73],
                'lower_right'  : [.5,.52,.89,.73],
            },
        }

        legtext_map = {
            'full' :  {'small' : .02, 'med' : .04, 'large' : .06},
            'upper' : {'small' : .02, 'med' : .04, 'large' : .06},
            'lower' : {'small' : .08, 'med' : .1,  'large' : .15},
        }
    
        # Resize legend
        if isinstance(pos, tuple):
            final_pos = pos
        else:
            if scale is not None:
                if 'right' in pos:
                    pos_map[option][pos][0] = pos_map[option][pos][2] - scale * (pos_map[option][pos][2] - pos_map[option][pos][0])
                elif 'left' in pos:
                    pos_map[option][pos][2] = pos_map[option][pos][0] + scale * (pos_map[option][pos][2] - pos_map[option][pos][0])
            final_pos = (pos_map[option][pos][0], pos_map[option][pos][1], pos_map[option][pos][2], pos_map[option][pos][3])

        if legtext_size: leg.SetTextSize(legtext_map[option][legtext_size])
        else: leg.SetTextSize(.04) if (option=='full' or option=='upper') else leg.SetTextSize(.1)

        leg.SetX1(final_pos[0])
        leg.SetX2(final_pos[2])
        leg.SetY1(final_pos[1])
        leg.SetY2(final_pos[3])

        gPad.Modified()
        leg.DrawClone()

    def createCanvas(self, option='hist', size=(800,800)):
        c = TCanvas('c', 'c', size[0], size[1])

        if option=='hist': 
            c.SetLeftMargin(0.15)
            c.SetRightMargin(0.05)
            c.SetBottomMargin(0.15)
            return c

        elif option=='ratio':
            pad1 = TPad('pad1', 'pad1', 0, 0.3, 1., 1.)
            pad1.SetBottomMargin(0.02) 
            pad1.SetLeftMargin(0.15)
            pad1.Draw()

            c.cd()  
            pad2 = TPad('pad2', 'pad2', 0, 0.05, 1, 0.3)
            pad2.SetTopMargin(0.033)  
            pad2.SetBottomMargin(0.5)
            pad2.SetLeftMargin(0.15)    

            pad2.SetFrameFillColor(0)
            pad2.SetFrameBorderMode(0)
            pad2.SetFrameFillColor(0)
            pad2.SetFrameBorderMode(0)
            pad2.Draw()
            return c, pad1, pad2
