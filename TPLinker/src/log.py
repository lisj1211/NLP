import logging


class TPLinkerLoging:
    """
    NER task logging
    """
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, log_name, log_file_path=None, level='info', fmt='%(message)s'):
        self.logger = logging.getLogger(log_name)
        self.logger.handlers.clear()

        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_dict.get(level, logging.INFO))

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)

        self.logger.addHandler(sh)

        if log_file_path is not None:
            fh = logging.FileHandler(log_file_path)
            fh.setFormatter(format_str)
            self.logger.addHandler(fh)
