"""Write synthetic Web Robots dumps for offline testing. Thin wrapper over ksai.synthetic."""

from __future__ import annotations

from ksai import config, synthetic

if __name__ == "__main__":
    synthetic.generate(config)
    print("Wrote synthetic dumps to data/raw and data/archive")