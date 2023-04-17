import logging
from pathlib import Path

class Logger:

    def __init__(self, name, log_dir, log_file):
        # ログ初期設定
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s]%(asctime)s %(name)s:%(message)s')
        # INFO以上はファイルに出力
        # 保存先ディレクトリがない場合は作成
        dir = Path(log_dir)
        dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir + '/' + log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        # ERROR以上はコンソールにも出力
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.ERROR)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, args, kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, args, kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, args, kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, args, kwargs)