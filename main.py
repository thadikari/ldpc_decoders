import numpy as np
import logging
import utils, codes
from models import models
import time


def test(args):
    log = logging.getLogger(args.channel + '|' + args.decoder)
    code = codes.get_code(args.code)
    code_n = code.get_n()
    x = code.parity_mtx[0] * 0 + args.codeword  # add 1 or 0
    model = models[args.channel]
    min_wec = args.min_wec
    saver = utils.Saver(args.data_dir, args.channel)

    for param in args.params:
        log.info('Evaluating Code: %s, Channel: %s, Decoder: %s, Parameter: %f' % (
            args.code, args.channel, args.decoder, param))
        run_id = [args.code, args.decoder, param]

        channel = model.Channel(param)
        decoder = getattr(model, args.decoder)(param, code)
        tot, wec, wer, bec, ber = 0, 0, 0., 0, 0.
        start_time = time.time()

        def log_status():
            log.info('Iter: %d, WEC: %d, WER: %f, BEC: %d, BER: %f'
                     % (tot, wec, wer, bec, ber))
            saver.add(run_id, wer)

        while wec < min_wec:
            if args.codeword == -1: x = code.cb[np.random.choice(code.cb.shape[0], 1)[0]]
            y = channel.send(x)
            x_hat = decoder.decode(y)
            errors = (~(x == x_hat)).sum()
            wec += errors > 0
            bec += errors
            tot += 1
            wer, ber = wec / tot, bec / (tot * code_n)
            if time.time() - start_time > args.log_freq:
                start_time = time.time()
                log_status()

        log_status()


def main():
    args = utils.setup_parser(codes.get_code_names(), models.keys(), ['ML', 'SPA']).parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.console:
        utils.setup_console_logger(log_level)
    else:
        utils.setup_file_logger(args.data_dir, 'test', log_level)

    np.random.seed(0)
    test(args)


if __name__ == "__main__":
    main()
