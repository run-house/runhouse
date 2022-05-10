from typer.testing import CliRunner

from runhouse import __app_name__, __version__, cli

runner = CliRunner()


def test_version():
    # run the application with the version
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    # app’s version is present in the standard output
    assert f"{__app_name__}=={__version__}\n" in result.stdout
