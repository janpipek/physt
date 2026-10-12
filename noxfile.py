import nox


@nox.session(python=["3.11", "3.12", "3.13", "3.14", "3.15"], venv_backend="uv")
def tests(session):
    session.install(".[all]")
    session.run("pytest")
