import datetime
import colorlog
import logging
import os


class Log:
    aliases = {
        logging.CRITICAL: ("critical", "crit", "c", "fatal"),
        logging.ERROR:    ("error", "err", "e"),
        logging.WARNING:  ("warning", "warn", "w"),
        logging.INFO:     ("info", "inf", "nfo", "i"),
        logging.DEBUG:    ("debug", "dbg", "d")
    }

    lvl = logging.DEBUG
    format_str = "%(log_color)s%(asctime)s | %(levelname)-8s | " \
                 "%(message)s (%(filename)s:%(lineno)d)%(reset)s"
    logging.root.setLevel(lvl)
    formatter = colorlog.ColoredFormatter(
            format_str,
            datefmt="%Y-%m-%d %H:%M:%S",
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'reset',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            })

    stream = logging.StreamHandler()
    stream.setLevel(lvl)
    stream.setFormatter(formatter)
    logger = logging.getLogger('eyesight')
    logger.setLevel(lvl)
    logger.addHandler(stream)

    crit = c = fatal = critical = logger.critical
    err = e = error = logger.error
    warning = w = warn = logger.warning
    inf = nfo = i = info = logger.info
    dbg = d = debug = logger.debug

    @classmethod
    def _parse_level(cls, lvl):
        for log_level in cls.aliases:
            if lvl == log_level or lvl in cls.aliases[log_level]:
                return log_level
        raise TypeError("Unrecognized logging level: %s" % lvl)

    @classmethod
    def level(cls, lvl=None):
        '''Get or set the logging level.'''
        if not lvl:
            return cls._lvl
        cls._lvl = cls._parse_level(lvl)
        cls.stream.setLevel(cls._lvl)
        logging.root.setLevel(cls._lvl)


log = Log()
if 'VERBOSE' in os.environ:
    log.level(os.environ['VERBOSE'])


class StopWatch:
    def __init__(self, desc, log_level='info'):
        self._desc = desc
        self._log_func = {
            logging.CRITICAL: log.crit,
            logging.ERROR: log.err,
            logging.WARNING: log.warn,
            logging.INFO: log.info,
            logging.DEBUG: log.debug
        }[Log._parse_level(log_level)]

    def __enter__(self):
        self._time_start = datetime.datetime.now()

    def __exit__(self, ex_type, ex_value, traceback):
        time_stop = datetime.datetime.now()
        self._log_func(self._desc + ' ' + str(time_stop - self._time_start))
