from src.validation import environment


def test_main_prints_validated_environment(monkeypatch, capsys):
    monkeypatch.setattr(
        environment,
        "validate_environment",
        lambda: {"region": "ap-southeast-1", "account": "123456789012"},
    )

    assert environment.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "region: ap-southeast-1",
        "account: 123456789012",
    ]
