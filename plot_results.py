#!/usr/bin/env python

import subprocess
import argparse

global args_

run = lambda ll: [print('>>', ll, flush=True), subprocess.run(ll, shell=True)]
exc = lambda e_, case: run(' '.join([e_] + case))
grph = lambda case: exc('python -u src/graph.py', case + args_.arg)
grph_ = lambda cases: [grph(case) for case in cases]
stat_ = lambda cases: [stat(case) for case in cases]

x_ = lambda a__: '--xlim ' + a__
y_ = lambda a__: '--ylim ' + a__
mi_ = lambda a__: '--max-iter=' + str(a__)


def main(args):
    global args_
    args_ = args
    for case in args.case: switch(case)


def switch(case):
    sv_ = lambda a__: '--file_name %s__%s' % (case, a__)
    fmt_str = '%s --and %s --error ber --legend decoder --title "%s, %s"'
    conf = lambda chl,cde: fmt_str%(chl,cde,chl.upper(),cde)

    if case == 'HMG':  # all hamming code sims
        co_ = lambda chl: conf(chl, '7_4_hamming')
        grph([co_('bec'), '--or_ ML SPA LP ADMM', sv_('BEC')])
        grph([co_('bsc'), '--or_ ML SPA MSA LP ADMM', sv_('BSC')])
        grph([co_('biawgn'), '--or_ ML SPA MSA LP ADMM', sv_('BIAWGN')])

    elif case == 'MAR':
        co_ = lambda chl: conf(chl, 'margulis')
        config = '--or_ ADMM'
        grph([co_('bec'), config, sv_('BEC')])
        grph([co_('bsc'), config, sv_('BSC')])
        grph([co_('biawgn'), config, sv_('BIAWGN')])


        '''
        Following three cases may not work with newest version of src/graph.py
        checkout commit e9f545908d27fee157f8896fcdf40939022708d5 for last working version
        '''

    elif case == 'REG_ENS':
        ens, code = '1200_3_6_rand_ldpc', '1200_3_6_ldpc'

        def plt_(chl, dec, args_en, args_cm, args_mi):
            prefix = chl + '_' + dec
            grph([chl, ens, dec, 'ensemble'] + args_en + [sv_(prefix + '_ensemble')])
            grph([chl, ens, dec, 'compare', sv_(prefix + '_compare'), '--extra ' + code] + args_cm)
            grph([chl, code, dec, 'max_iter', sv_(prefix + '_max_iter')] + args_mi)

        plt_('bec', 'SPA', ['--xlim .3 .5 --ylim 2e-7 .5 --max-iter=10'],
             ['--max-iter=10 --xlim .3 .5 --ylim 3e-5 .5'], [''])
        plt_('bsc', 'MSA', ['--xlim 0.02 0.08 --ylim 6e-6 .2 --max-iter=10'],
             ['--max-iter=10 --xlim 0.015 0.08'], [''])
        plt_('biawgn', 'MSA', ['--xlim .5 3 --ylim 3e-5 .2 --max-iter=10'],
             ['--max-iter=10 --xlim .5 3 --ylim 3e-5 .2'], ['--xlim .5 3 --ylim 4e-4 .2'])
        plt_('bsc', 'SPA', ['--max-iter=10'], ['--max-iter=10'], [''])
        plt_('biawgn', 'SPA', ['--xlim .5 3 --max-iter=10'],
             ['--max-iter=10 --xlim .5 3'], ['--xlim .5 3 --ylim 3e-5 .2'])

        grph(['bsc', code, 'SPA MSA', 'comp_dec', mi_(10), sv_('BSC_comp_dec')])
        grph(['biawgn', code, 'SPA MSA', 'comp_dec', mi_(10), x_('.5 2.75'), sv_('BIAWGN_comp_dec')])

    elif case == 'IREG_ENS':
        ens = '1200_rho_x5_rand_ldpc'

        def plt_(chl, dec, args_en):
            prefix = chl + '_' + dec
            grph([chl, ens, dec, 'ensemble'] + args_en + [sv_(prefix + '_ensemble')])

        plt_('bec', 'SPA', ['--xlim .3 .5 --ylim 2e-7 .5 --max-iter=10'])
        plt_('bsc', 'MSA', ['--xlim 0.02 0.08 --ylim 6e-6 .2 --max-iter=10'])
        plt_('biawgn', 'MSA', ['--xlim .5 3 --ylim 3e-5 .2 --max-iter=10'])
        plt_('bsc', 'SPA', ['--max-iter=10'])
        plt_('biawgn', 'SPA', ['--xlim .5 3 --max-iter=10'])

        grph(['bsc', ens, 'SPA MSA', 'comp_dec', mi_(10), sv_('BSC_comp_dec')])
        grph(['biawgn', ens, 'SPA MSA', 'comp_dec', mi_(10), x_('.5 2.75'), sv_('BIAWGN_comp_dec')])

    elif case == 'COMP_REG_IREG':
        reg, irg = '1200_3_6_rand_ldpc', '1200_rho_x5_rand_ldpc'
        co__ = lambda cnl, dec: ['%s %s %s compare --extra %s' % (cnl, irg, dec, reg),
                                 sv_(cnl + '_' + dec + '_compare')]
        grph(co__('bec', 'SPA'))
        grph(co__('bsc', 'MSA') + ['--xlim .015 0.08'])
        grph(co__('biawgn', 'MSA'))
        grph(co__('bsc', 'SPA'))
        grph(co__('biawgn', 'SPA'))

        cd__ = lambda cnl, dec: ['%s %s %s comp_dec --extra %s' % (cnl, irg, dec, reg),
                                 sv_(cnl + '_' + 'comp_dec')]
        grph(cd__('bsc', 'MSA SPA') + ['--xlim .015 0.08'])
        grph(cd__('biawgn', 'MSA SPA'))


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', nargs='+', help='specify case(s)')
    parser.add_argument('arg', help='arguments passed to graph.py', default=[], nargs=argparse.REMAINDER)
    return parser


if __name__ == '__main__':
    main(setup_parser().parse_args())
