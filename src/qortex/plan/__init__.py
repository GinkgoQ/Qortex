from qortex.plan.selector import ESSENTIAL_FILENAMES, Selector

__all__ = ["DownloadPlanner", "Selector", "LockFile", "ESSENTIAL_FILENAMES"]


def __getattr__(name: str):
    if name == "DownloadPlanner":
        from qortex.plan.planner import DownloadPlanner

        return DownloadPlanner
    if name == "LockFile":
        from qortex.plan.lock import LockFile

        return LockFile
    raise AttributeError(name)
