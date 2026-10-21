"""Dashboard Generator Agent.

``run`` est importé à la demande pour éviter un import circulaire
(agents.tools.dashboard -> agents.dashboard.* -> agent -> agents.tools.dashboard).
"""

from agents.dashboard.models import DashboardOptions, DashboardReport, DashboardRequest

__all__ = ["run", "DashboardOptions", "DashboardReport", "DashboardRequest"]


def __getattr__(name: str):
    if name == "run":
        from agents.dashboard.agent import run
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")