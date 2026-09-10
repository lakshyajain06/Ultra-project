"""Isaac Teleop controller graph. Poses leave this graph in simulation world XYZW."""

import numpy as np
from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeting_engine.interface import (
    BaseRetargeter, OptionalType, OutputCombiner, TensorGroupType, ValueInput,
)
from isaacteleop.retargeting_engine.tensor_types import (
    ControllerInput, ControllerInputIndex as I, DLDataType, NDArrayType, TransformMatrix,
)


class PackControllers(BaseRetargeter):
    """Two rows: pose, trigger, squeeze, valid, buttons, and thumbstick XY.

    Invalid controllers retain button state when available, but never pose control.
    This is device input, not the robot's 22-dimensional action.
    """

    def input_spec(self):
        return {side: OptionalType(ControllerInput()) for side in ("left", "right")}

    def output_spec(self):
        return {"action": TensorGroupType("action", [NDArrayType(
            "controllers", shape=(30,), dtype=DLDataType.FLOAT, dtype_bits=32,
        )])}

    def _compute_fn(self, inputs, outputs, context):
        packet = np.zeros((2, 15), dtype=np.float32)
        for n, side in enumerate(("left", "right")):
            group = inputs[side]
            if group.is_none:
                continue
            packet[n, :3] = np.asarray(group[I.GRIP_POSITION])
            packet[n, 3:7] = np.asarray(group[I.GRIP_ORIENTATION])
            packet[n, 7:] = [float(group[index]) for index in (
                I.TRIGGER_VALUE, I.SQUEEZE_VALUE, I.GRIP_IS_VALID,
                I.PRIMARY_CLICK, I.SECONDARY_CLICK, I.THUMBSTICK_CLICK,
                I.THUMBSTICK_X, I.THUMBSTICK_Y,
            )]
        outputs["action"][0] = packet.ravel()


def build_pipeline():
    controllers = ControllersSource("ultra_controllers")
    anchor = ValueInput("world_T_anchor", TransformMatrix())
    world = controllers.transformed(anchor.output(ValueInput.VALUE))
    packed = PackControllers("ultra_packet").connect({
        "left": world.output(ControllersSource.LEFT),
        "right": world.output(ControllersSource.RIGHT),
    })
    return OutputCombiner({"action": packed.output("action")})
