from SCAutolib import logger
from contextlib import contextmanager
import re

SECURE_LOG = '/var/log/secure'


def sc_successful_log(username: str):
    string = (
        r'.* localhost gdm-smartcard\]\[[0-9]+\]: '
        r'pam_sss\(gdm-smartcard:auth\): authentication success;'
        r'.*user=' + username + r'@shadowutils.*'
    )

    return string


@contextmanager
def assert_log(filename: str, expected_log: str):
    logger.info(f'Opening log file {filename}')
    with open(filename) as f:
        # Move file pointer to the end of file
        f.seek(0, 2)
        p = re.compile(expected_log)

        # Run the actual actions
        try:
            yield

        finally:
            logger.info(f'Asserting regex `{expected_log}` in {filename}')
            for line in f:
                m = p.match(line)
                if m:
                    # found the log
                    text = m.group()
                    logger.info(f'Found matching line: {text}')
                    return

            raise Exception('The log was not found.')
