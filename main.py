import numpy as np
import logging
import utils, codes
from models import models


def test(args):
    log = logging.getLogger(args.channel + '|' + args.decoder)
    code = codes.get_code(args.code)
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
        tot, wec, wer = 0, 0, 0.
        log_status = lambda: (log.info('WEC: %d, Iter: %d, WER: %f' % (wec, tot, wer)), saver.add(run_id, wer))
        while wec < min_wec:
            if args.codeword == -1: x = code.cb[np.random.choice(code.cb.shape[0], 1)[0]]
            y = channel.send(x)
            x_hat = decoder.decode(y)
            wec += ~(x == x_hat).all()
            tot += 1
            wer = wec / tot
            if tot % 1e6 == 0: log_status()

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
