#!/usr/bin/env python

import subprocess
import argparse
import src.utilities.misc


run = lambda ll: [print('>>', ll, flush=True), subprocess.run(ll, shell=True)]
exc = lambda e_, case: run(' '.join([e_] + case))
grph = lambda case: exc('python -u src/graph.py', case + args_.arg)
grph_ = lambda cases: [grph(case) for case in cases]
stat_ = lambda cases: [stat(case) for case in cases]

x_ = lambda a__: '--xlim ' + a__
y_ = lambda a__: '--ylim ' + a__


def main():
    for case in args_.case:
        sv_ = lambda a__: '--file_name %s__%s' % (case, a__)
        all_cases.get(case)(sv_)


all_cases = src.utilities.misc.Registry()
reg_case = all_cases.reg

fmt_str = '--and %s-%s --error ber --legend decoder --title "%s, %s"'
conf = lambda chl,cde: fmt_str%(chl,cde,chl.upper(),cde)

@reg_case
def HMG(sv_):  # all hamming code sims
    co_ = lambda chl: conf(chl, '7_4_hamming')
    grph([co_('bec'), '--or_ ML SPA LP ADMM', sv_('BEC')])
    grph([co_('bsc'), '--or_ ML SPA MSA LP ADMM', sv_('BSC')])
    grph([co_('biawgn'), '--or_ ML SPA MSA LP ADMM', sv_('BIAWGN')])

@reg_case
def MAR(sv_):
    co_ = lambda chl: conf(chl, 'margulis')
    config = '--or_ ADMM --error wer'
    grph([co_('bec'), config, sv_('BEC')])
    grph([co_('bsc'), config, sv_('BSC')])
    grph([co_('biawgn'), config, sv_('BIAWGN')])


def plt_ens(ens, prefix, args_en, sv_, chl, CHL, dec):
    ens_kw = f'--and {chl}-{ens} {dec} 10.json --title "{CHL}, {dec} decoder, {ens} ensemble"'
    grph([ens_kw, '--type ensemble'] + args_en + [sv_(prefix + '_ensemble')])

@reg_case
def REG_ENS(sv_):
    ens, code = '1200_3_6_rand_ldpc', '1200_3_6_ldpc'

    def plt_(chl, dec, args_en, args_cm, args_mi):
        CHL = chl.upper()
        prefix = chl + '_' + dec
        plt_ens(ens, prefix, args_en, sv_, chl, CHL, dec)
        comp_kw = f'--or_ {ens} {code} --and {chl} {dec} 10.json --title "{CHL}, {dec} decoder, {ens} ensemble" --type regex_average --group "{ens}_[0-9]+-{dec}" "ldpc_rand average"'
        grph([comp_kw, sv_(prefix + '_compare')] + args_cm)
        max_kw = f'--and {chl}-{code} {dec} --title "{CHL}, {code}, {dec} decoder, Effect of iterations cap"'
        grph([max_kw, sv_(prefix + '_max_iter')] + args_mi)

    plt_('bsc', 'MSA', ['--xlim 0.02 0.08 --ylim 6e-6 .2'],
         ['--xlim 0.015 0.08'], [''])
    plt_('biawgn', 'MSA', ['--xlim .5 3 --ylim 3e-5 .2'],
         ['--xlim .5 3 --ylim 3e-5 .2'], ['--xlim .5 3 --ylim 4e-4 .2'])

    plt_('bec', 'SPA', ['--xlim .3 .5 --ylim 2e-7 .5'],
         ['--xlim .3 .5 --ylim 3e-5 .5'], [''])
    plt_('bsc', 'SPA', [], [], [])
    plt_('biawgn', 'SPA', ['--xlim .5 3'],
         ['--xlim .5 3'], ['--xlim .5 3 --ylim 3e-5 .2'])

    fmt_str_ens = '--and %s-%s 10.json --or_ SPA MSA --legend decoder --title "%s, %s ensemble, Average performance"'
    conf_ens = lambda chl: fmt_str_ens%(chl,code,chl.upper(),code)
    grph([conf_ens('bsc'), sv_('BSC_comp_dec')])
    grph([conf_ens('biawgn'), x_('.5 2.75'), sv_('BIAWGN_comp_dec')])

@reg_case
def IREG_ENS(sv_):
    ens = '1200_rho_x5_rand_ldpc'

    def plt_(chl, dec, args_en):
        plt_ens(ens, chl + '_' + dec, args_en, sv_, chl, chl.upper(), dec)

    plt_('bec', 'SPA', ['--xlim .3 .5 --ylim 2e-7 .5'])
    plt_('bsc', 'MSA', ['--xlim 0.02 0.08 --ylim 6e-6 .2'])
    plt_('biawgn', 'MSA', ['--xlim .5 3 --ylim 3e-5 .2'])
    plt_('bsc', 'SPA', [])
    plt_('biawgn', 'SPA', ['--xlim .5 3'])

    group_ = lambda dec: f'--group "{ens}_[0-9]+-{dec}" {dec}'
    fmt_str_ens = '--and %s-%s 10.json --or_ SPA MSA --type regex_average --title "%s, %s ensemble, Average performance" ' + group_('SPA') + ' ' + group_('MSA')
    conf_ens = lambda chl: fmt_str_ens%(chl,ens,chl.upper(),ens)
    grph([conf_ens('bsc'), sv_('BSC_comp_dec')])
    grph([conf_ens('biawgn'), x_('.5 2.75'), sv_('BIAWGN_comp_dec')])

@reg_case
def COMP_REG_IREG(sv_):
    reg, irg = '1200_3_6_rand_ldpc', '1200_rho_x5_rand_ldpc'
    group_ = lambda cde,leg: f'--group "{cde}_[0-9]+" {leg}'

    co__ = lambda cnl, dec: ['--and %s %s --or_ %s %s' % (cnl, dec, irg, reg),
                             '--type regex_average --title "%s, %s decoder, Average performance of ensemble"'%(cnl.upper(), dec),
                             group_(reg, reg), group_(irg, irg),
                             sv_(cnl + '_' + dec + '_compare')]
    grph(co__('bec', 'SPA'))
    grph(co__('bsc', 'MSA') + ['--xlim .015 0.08'])
    grph(co__('biawgn', 'MSA'))
    grph(co__('bsc', 'SPA'))
    grph(co__('biawgn', 'SPA'))

    group_ = lambda cde,dec,leg: f'--group "{cde}_[0-9]+-{dec}" {leg}'
    cd__ = lambda cnl, dec: ['--and %s --or_ %s %s' % (cnl, irg, reg),
                             '--type regex_average --title "%s, Average performance of ensemble"'%cnl.upper(),
                             group_(reg,'SPA',f'SPA-{reg}'),
                             group_(reg,'MSA',f'MSA-{reg}'),
                             group_(irg,'SPA',f'SPA-{irg}'),
                             group_(irg,'MSA',f'MSA-{irg}'),
                             sv_(cnl + '_' + 'comp_dec')]
    grph(cd__('bsc', 'MSA SPA') + ['--xlim .015 0.08'])
    grph(cd__('biawgn', 'MSA SPA'))


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', nargs='+', help='specify case(s)', choices=all_cases.keys(), default=all_cases.keys())
    parser.add_argument('arg', help='arguments passed to graph.py', default=[], nargs=argparse.REMAINDER)
    return parser


if __name__ == '__main__':
    args_ = setup_parser().parse_args()
    main()
