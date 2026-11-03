"""CyberEther adapters: one class per Python block, each wrapping a stage (or the source/sink of the graph).

The counterpart of :mod:`kremetart.operators` for the CyberEther runner. A block is built from a plain,
JSON-serialisable config on its first compute cycle and then called once per cycle with the block's
``ctx``; the flowgraph YAML (:mod:`kremetart.core.cyberether_runner`) holds only a stub that does this.
"""
