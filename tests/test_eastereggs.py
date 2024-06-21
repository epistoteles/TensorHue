from tensorhue.eastereggs import pride


def test_pride_output(capsys):
    pride()
    captured = capsys.readouterr()
    out = captured.out.rstrip("\n")
    assert len(out.split("\n")) == 3
    assert out.count("▀") == 30
