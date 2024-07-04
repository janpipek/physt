"""Script showing some of the physt examples.

You can run this like:

    python -m physt.examples
"""
from rich.pretty import pprint

from physt.examples import normal_h1, normal_h2

h1 = normal_h1()
h2 = normal_h2()

print("A 1D histogram: ")
pprint(h1)

print()
print("Its stats:")
pprint(h1.statistics)

print()
h1.plot(backend="ascii", show_values=True)

print("A 2D histogram: ")
pprint(h2)

h2.plot(backend="ascii")
