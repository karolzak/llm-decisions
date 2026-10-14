import enum
from types import SimpleNamespace

from edb.runners.aml import TERMINAL, status_of


class FakeStatus(enum.Enum):
    FAILED = "Failed"

    def __str__(self):
        return f"JobStatus.{self.name}"


class OpaqueStatus:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


def test_status_normalisation():
    assert status_of(SimpleNamespace(status="Completed")) == "Completed"
    assert status_of(SimpleNamespace(status=FakeStatus.FAILED)) == "Failed"
    assert status_of(SimpleNamespace(status=OpaqueStatus("JobStatus.NOT_RESPONDING"))) == \
        "NotResponding"
    assert {"Failed", "NotResponding"} <= TERMINAL

def test_job_key():
    from edb.runners.aml import job_key
    assert job_key("c2_oss_sequence_scoring", "short") == "c2_oss_sequence_scoring__short"
