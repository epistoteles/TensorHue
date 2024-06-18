import pytest
from tensorhue.eastereggs import pride


def test_pride_output(capsys):
    pride()
    captured = capsys.readouterr()
    assert len(captured.out.split("\n")) == 5
    assert captured.out.count("▀") == 30
