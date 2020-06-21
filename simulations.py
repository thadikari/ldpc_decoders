#!/usr/bin/env python

import argparse
import src.utilities.misc


prt = lambda ss: print(' '.join(ss + args_.arg), flush=True)
exc_cases = lambda cases: [prt(case) for case in cases]
exc_ens = lambda prefix, count: [exc_def_cases('%s_%d' % (prefix, i + 1)) for i in range(count)]

p_ = lambda a__: '--params ' + a__
cw_ = lambda a__: '--codeword=' + str(a__)
mi_ = lambda a__: '--max-iter=' + str(a__)
mw_ = lambda a__: '--min-wec=' + str(a__)
sp_ = lambda ll: p_(' '.join(['%g' % val for val in ll]))
stp = lambda init, step, count: [init + cnt * step for cnt in range(count)]


def stps(init, steps):
    last, ll = init, []
    for step, count in steps:
        ll += stp(last, step, count)
        last = ll[-1]
    return ll


def exc_def_cases(code, mi=10, mw=100):
    cases = [
        ['bec', code, 'SPA', cw_(0), mi_(mi), mw_(mw), p_('.5 .475 .45 .425 .4 .375 .35 .34 .33 .325 .32 .31 .3')],
        ['bsc', code, 'MSA', cw_(1), mi_(mi), mw_(mw),
         p_('.081 .0751 .071 .0651 .061 .0551 .051 .0451 .041 .0351 .031 .0251 .021 .0151 .01')],
        ['biawgn', code, 'MSA', cw_(1), mi_(mi), mw_(mw),
         p_('.5 .75 1. 1.25 1.5 1.75 2. 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0')],
        # sp_(stps(.5, [(.25, 7), (.1, 10)]))],
        ['bsc', code, 'SPA', cw_(0), mi_(mi), mw_(mw), sp_(stp(.1, -.01, 7))],
        ['biawgn', code, 'SPA', cw_(0), mi_(mi), mw_(mw), p_('.5 .75 1. 1.25 1.5 1.75 2. 2.25 2.5 2.75 3.')]
    ]

    exc_cases(cases)


def main():
    for case in args_.case: all_cases.get(case)()


all_cases = src.utilities.misc.Registry()
reg_case = all_cases.reg

@reg_case
def HMG():  # all hamming code sims
    p_bec = '.5 .4 .3 .2 .1 .08 .06 .04 .02'
    p_bsc = p_bec + ' .25 .15 .01 .008 .006 .004 .002'

    decs_bec = ['ML', 'LP', 'SPA', 'ADMM']
    decs_def = ['ML', 'LP', 'SPA', 'MSA', 'ADMM']

    code, config = '7_4_hamming', [cw_(1), mw_(300)]
    cases = [['bec', code, dec, p_(p_bec)] + config for dec in decs_bec] + \
            [['bsc', code, dec, p_(p_bsc)] + config for dec in decs_def] + \
            [['biawgn', code, dec, sp_(stp(2, .5, 11))] + config for dec in decs_def]
    exc_cases(cases)

@reg_case
def MAR():  # margulis code sims
    code, config = 'margulis', [cw_(1), mw_(100)]
    cases = [
        ['bec', code, 'ADMM', p_('.5 .475 .45 .425 .4 .375 .35 .34 .33 .325 .32 .31 .3')] + config,
        ['bsc', code, 'ADMM', p_('.1 .09 .08 .07 .06 .05 .04')] + config,
        ['biawgn', code, 'ADMM', p_('.5 .75 1. 1.25 1.5 1.75 2. 2.25 2.5 2.75 3.0')] + config
    ]
    exc_cases(cases)
    exc_def_cases(code)

@reg_case
def REG_BAD():
    exc_def_cases('1200_3_6_ldpc')
    for mi in [0, 1, 2, 3, 6, 40, 100]: exc_def_cases('1200_3_6_ldpc', mi)

@reg_case
def REG_ENS():
    exc_ens('1200_3_6_rand_ldpc', 10)

@reg_case
def IREG_ENS():
    exc_ens('1200_rho_x5_rand_ldpc', 10)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', nargs='+', help='specify case(s)')
    parser.add_argument('arg', help='arguments passed to wrapped command', nargs=argparse.REMAINDER)
    return parser


if __name__ == '__main__':
    args_ = setup_parser().parse_args()
    main()
