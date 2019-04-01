import argparse
import logging
import os
import re

from models import models
import utils


def is_valid(data, args, pattern):
    if data is None: return 0
    if data.get('channel', '') != args.channel: return 0
    if data.get('decoder', '') not in args.decoder: return 0
    return pattern.match(data.get('code', ''))


def main(args):
    log = logging.getLogger()
    errors = ['wer', 'ber']
    file_list = utils.get_data_file_list(args.data_dir)
    matches = dict(((err, []) for err in errors))
    pattern = re.compile('^' + args.prefix + '_[0-9]+$')
    log.info('regex to match: ' + pattern.pattern)

    src_list = []
    for file_name in file_list:
        data = utils.load_json(os.path.join(args.data_dir, file_name))
        if is_valid(data, args, pattern):
            log.info('found match: %s' % file_name)
            src_list.append(data['code'])
            for err in errors: matches[err].append(data[err])

    avg = {}
    for err in errors:
        ll = {}
        for inst in matches[err]:
            for point in inst:
                if point not in ll.keys(): ll[point] = []
                ll[point].append(inst[point])

        for point in ll:
            val = ll[point]
            ll[point] = sum(val) / float(len(val))

        avg[err] = ll

    id_keys = ('type', 'channel', 'prefix', 'decoder')
    id_val = ('stats', args.channel, args.prefix, args.decoder)
    saver = utils.Saver(args.data_dir, list(zip(id_keys, id_val)))
    saver.add_meta('sources', src_list)
    saver.add_all(avg)
    log.info('Done!')


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('channel', help='channel', choices=models.keys())
    parser.add_argument('prefix', help='prefix of code')
    parser.add_argument('decoder', help='decoder', choices=utils.decoder_names)
    return utils.bind_parser_common(parser)


if __name__ == "__main__":
    utils.setup_console_logger()
    main(setup_parser().parse_args())
