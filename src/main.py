import numpy as np
import logging
import utils, codes
from models import models
import time
from collections import OrderedDict


def test(args):
    model = models[args.channel]
    dec_fac = getattr(model, args.decoder)
    id_keys = ['channel', 'code', 'decoder', 'codeword'] + dec_fac.id_keys
    id_val = tuple(str(vars(args)[key]) for key in id_keys)
    log = logging.getLogger('.'.join(id_val))
    code = codes.get_code(args.code)
    code_n = code.get_n()
    x = code.parity_mtx[0] * 0 + args.codeword  # add 1 or 0
    min_wec = args.min_wec
    saver = utils.Saver(args.data_dir, list(zip(id_keys, id_val)))

    for param in args.params:
        log.info('Starting parameter: %f' % param)

        channel = model.Channel(param)
        decoder = dec_fac(param, code, **vars(args))
        tot, wec, wer, bec, ber = 0, 0, 0., 0, 0.
        start_time = time.time()

        def log_status():
            val_dict = OrderedDict(zip(('tot', 'wec', 'wer', 'bec', 'ber'),
                                       (tot, wec, wer, bec, ber)))
            log.info(', '.join(('%s:%s' % (key.upper(), val_dict[key]) for key in val_dict)))
            saver.add(param, val_dict)

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
    log.info('Done!')


def main():
    args = utils.setup_parser(codes.get_code_names(), models.keys(), utils.decoder_names).parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.console:
        utils.setup_console_logger(log_level)
    else:
        utils.setup_file_logger(args.data_dir, 'test', log_level)

    np.random.seed(0)
    test(args)


if __name__ == "__main__":
    main()
