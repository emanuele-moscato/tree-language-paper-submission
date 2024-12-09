import logging


class Singleton(type):
  def __call__(cls, *args, **kwargs):
    if not hasattr(cls, 'instance'):
        cls.instance = super(Singleton, cls).__call__(*args, **kwargs)

    return cls.instance



class SingletonLogger(object, metaclass=Singleton):
    """
    Implementation of a logger as a singleton class.
    """
    def __init__(self, name, level=logging.DEBUG, log_file_path=None):
        # Create logger object.
        self._logger = logging.getLogger(name)

        # Set logging level.
        self._logger.setLevel(level)

        # Create formatter.
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create console handler, set its level and add the formatter to it.
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)

        # Add console handler to the logger.
        self._logger.addHandler(ch)

        if log_file_path is not None:
            # Create file handler and set its level.
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(level)

            # Add formatter to the file handler.
            fh.setFormatter(formatter)

            # Add the file handler to the logger.
            self._logger.addHandler(fh)


def get_logger(name, level=logging.DEBUG, log_file_path=None):
    """
    Returns a SingletonLogger object. The object is instantiated only once at
    the first call; any subsequent calls return the previously created
    instance.
    """
    return SingletonLogger(name, level, log_file_path)._logger