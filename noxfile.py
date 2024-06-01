"""Manages Nox sessions used for automated testing."""

# Pip
from nox import Session, session


@session(python=["3.11", "3.12"])  # type: ignore[misc]
def tests(test_session: Session) -> None:
    """Run the tests with coverage.

    Args:
        test_session: The Nox session to run the tests with.
    """
    test_session.install(".", "pytest-cov", "setuptools", "wheel")
    test_session.run("python", "-m", "build", "--cpp")
    test_session.run("pytest", "--cov-append")
    test_session.run("coverage", "lcov")
