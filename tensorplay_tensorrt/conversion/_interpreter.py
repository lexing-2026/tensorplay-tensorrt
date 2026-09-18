"""Node-by-node translation of one captured graph into TRT layers.

The interpreter walks the graph in order, keeping every produced ITensor in
an environment keyed by node name.  Each call node resolves its converter by
op name, receives the evaluated operands, and returns the layer's output;
the node's consumers then reference that value through the environment.
Placeholder nodes declare the network inputs from the sample tensors; the
output node marks the network output.
"""

from __future__ import annotations

from typing import Any

from ._converter_registry import CONVERTERS, UnsupportedOperator, target_name
from ._conversion_context import ConversionContext
from .converter_utils import trt_dtype


class Interpreter:
    """Translate one captured graph into layers of one TRT network."""

    def __init__(
        self,
        network: Any,
        example_inputs: list[Any],
        settings: Any,
        trt: Any,
    ) -> None:
        self.ctx = ConversionContext(network, settings, trt)
        self.example_inputs = list(example_inputs)
        self.env: dict[str, Any] = {}
        self._input_index = 0

    def run(self, graph_module: Any) -> Any:
        """Translate the whole graph; returns the marked output ITensor."""

        output_value = None
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                self.env[node.name] = self._placeholder(node)
            elif node.op == "output":
                output_value = self._output(node, graph_module)
            elif node.op in ("call_function", "call_method"):
                self.env[node.name] = self._call(node)
            else:
                raise UnsupportedOperator(
                    f"unsupported node kind {node.op!r} (node {node.name!r})"
                )
        if output_value is None:
            raise ValueError("graph produced no output")
        return output_value

    def _placeholder(self, node: Any) -> Any:
        if self._input_index >= len(self.example_inputs):
            raise ValueError(f"placeholder {node.name!r} has no example input")
        sample = self.example_inputs[self._input_index]
        self._input_index += 1
        shape = tuple(int(dim) for dim in sample.shape)
        return self.ctx.net.add_input(
            node.name, trt_dtype(self.ctx.trt, sample.dtype), shape
        )

    def _fetch(self, value: Any) -> Any:
        """Evaluate one node argument: a produced value or a literal.

        Arguments arrive as graph references (nodes), not values; the
        interpreter is the piece that turns each reference into the ITensor
        its producer emitted.  Everything else -- Python numbers, strings,
        containers -- passes through untouched for the converter to judge.
        """

        if isinstance(value, list):
            return [self._fetch(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._fetch(item) for item in value)
        key = getattr(value, "name", None)
        if key is not None and key in self.env:
            return self.env[key]
        return value

    def _call(self, node: Any) -> Any:
        name = target_name(node)
        converter = CONVERTERS.get(name)
        if converter is None:
            raise UnsupportedOperator(
                f"conversion of operation {name!r} (node {node.name!r}) "
                "is not supported"
            )
        self.ctx.current_node_name = node.name
        args = [self._fetch(arg) for arg in node.args]
        kwargs = {key: self._fetch(value) for key, value in node.kwargs.items()}
        return converter(self.ctx, node.target, args, kwargs, node.name)

    def _output(self, node: Any, graph_module: Any) -> Any:
        values = node.args[0]
        if isinstance(values, (list, tuple)):
            if len(values) != 1:
                raise ValueError("multi-output regions are not supported yet")
            values = values[0]
        key = getattr(values, "name", None)
        if key not in self.env:
            raise ValueError(f"output {key!r} was never produced")
        tensor = self.env[key]
        self.ctx.net.mark_output(tensor)
        return tensor
