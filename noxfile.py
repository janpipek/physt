import nox


@nox.session(python=["3.9", "3.10", "3.11", "3.12"], venv_backend="uv")
def tests(session):
    session.install(".[all]")
    session.run("pytest")
