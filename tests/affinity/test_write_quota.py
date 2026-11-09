import pytest
from kfold.affinity.write_quota import fresh_write_quota


def test_only_the_write_mount_is_queried_once():
    calls=[]
    def run(args):
        calls.append(args)
        return '/scratch 1 10000000000 10000000000 - 1 5000000 5000000 -\n'
    result=fresh_write_quota(run=run)
    assert list(result)==['/scratch'] and len(calls)==1
    assert calls[0]==['lfs','quota','-g','bs0247','-q','/scratch']


@pytest.mark.parametrize('text',['/appsdata 1 10000000000 10000000000 - 1 5000000 5000000 -',
    '/scratch 9999999999 10000000000 10000000000 - 1 5000000 5000000 -',
    '/scratch 1 10000000000 10000000000 - 4999999 5000000 5000000 -', 'broken'])
def test_unknown_mount_low_space_low_files_and_bad_response_fail_closed(text):
    with pytest.raises((ValueError,RuntimeError)):fresh_write_quota(run=lambda args:text)


def test_query_error_is_not_retried():
    calls=[]
    def fail(args):calls.append(args);raise TimeoutError('quota timeout')
    with pytest.raises(TimeoutError):fresh_write_quota(run=fail)
    assert len(calls)==1
