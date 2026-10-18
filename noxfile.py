import nox


@nox.session(python=["3.11", "3.12", "3.13", "3.14", "3.15"], venv_backend="uv")
def tests(session):
    session.install(".[all]")
    session.install("pytest")
    session.run("pytest")

@nox.session(python=[ "3.14t", "3.15t"], venv_backend="uv")
def tests_free_threading(session):
    session.install(".[all-ft]")
    session.install("pytest")
    session.run("pytest")
