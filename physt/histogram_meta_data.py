from collections import UserDict


class HistogramMetaData(UserDict):
    """Collection of histogram meta-data.

    This object works as a dictionary with arbitrary keys
    but provides some properties that have special meaning.

    Attributes
    ----------
    - title

    """
    @property
    def title(self) -> str:
        """Title to be displayed at the top of the histogram."""
        return self.get("title", None)

    @title.setter
    def title(self, value):
        self.data["title"] = str(value)

    