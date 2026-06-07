def test_import():
    import kremetart

    assert hasattr(kremetart, "__version__")


def test_version_is_string():
    from kremetart import __version__

    assert isinstance(__version__, str)
