"""Script showing some of the physt examples.

You can run this like:

    python -m physt.examples
"""
import importlib
import sys

from rich.json import JSON
from rich.panel import Panel

if not importlib.util.find_spec("rich"):
    print("Please, install rich or physt[terminal] to view nice terminal output.")
    sys.exit(-1)

from rich.console import Console

from physt.examples import normal_h1, normal_h2

console = Console()


h1 = normal_h1()
h2 = normal_h2()

console.print("[bold]A 1D histogram:[/bold]")
console.print(h1)

console.print()
console.print("[bold]The associated stats:[/bold]")
console.print(h1.statistics)

console.print()
console.print("[bold]And the plot:[/bold]")

h1.plot(backend="ascii", show_values=True)
console.print()

console.print("[bold]JSON fully describing the histogram:[/bold]")
console.print(Panel(JSON(h1.to_json()), width=min(console.width, 60)))

console.print("[bold]A 2D histogram:[/bold]")
console.print(h2)

console.print()
console.print("[bold]And the plot:[/bold]")
h2.plot(backend="ascii")
