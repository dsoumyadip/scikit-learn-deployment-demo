import logging
import os.path

from demo.constants import LOG_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

fh = logging.FileHandler(filename=os.path.join(LOG_DIR, 'scikit-learn-demo.log'))

sh = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(sh)
