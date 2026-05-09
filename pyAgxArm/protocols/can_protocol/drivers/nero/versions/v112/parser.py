from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type

from .......utiles.numeric_codec import NumericCodec as nc, DEG2RAD
from .....msgs.core.msg_abstract import MessageAbstract
from .....msgs.piper.default import (
    ArmMsgJointCtrl12,
    ArmMsgJointCtrl34,
    ArmMsgJointCtrl56,
)
from .....msgs.nero.default import ArmMsgJointCtrl7
from ..v111.parser import Codec as V111Codec, Parser as V111Parser


class Codec(V111Codec):
    """Nero v112 codec."""

    def decode_170_joint_ctrl_7(self, m: ArmMsgJointCtrl7, d: bytearray) -> None:
        m.joint_7 = (
            nc.ConvertToNegative_32bit(nc.ConvertBytesToInt(d, 0, 4))
            * 1e-3
            * DEG2RAD
        )


class Parser(V111Parser):
    """Nero v112 parser."""

    if TYPE_CHECKING:
        leader_joint_12: Optional[MessageAbstract[ArmMsgJointCtrl12]]
        leader_joint_34: Optional[MessageAbstract[ArmMsgJointCtrl34]]
        leader_joint_56: Optional[MessageAbstract[ArmMsgJointCtrl56]]
        leader_joint_7: Optional[MessageAbstract[ArmMsgJointCtrl7]]

    def __init__(self, fps_manager, codec: Optional[Codec] = None):
        super().__init__(fps_manager, codec=codec or Codec())
        self._codec = codec or Codec()

    def _build_rx_map(
        self,
    ) -> Dict[int, Tuple[str, Type, Callable[[object, bytearray], None]]]:
        rx = super()._build_rx_map()
        for can_id in (0x501, 0x502, 0x503, 0x504, 0x505, 0x506, 0x507):
            rx.pop(can_id, None)
        rx.update(
            {
                0x155: (
                    "leader_joint_12",
                    ArmMsgJointCtrl12,
                    self._codec.decode_155_joint_ctrl_12,
                ),
                0x156: (
                    "leader_joint_34",
                    ArmMsgJointCtrl34,
                    self._codec.decode_156_joint_ctrl_34,
                ),
                0x157: (
                    "leader_joint_56",
                    ArmMsgJointCtrl56,
                    self._codec.decode_157_joint_ctrl_56,
                ),
                0x170: (
                    "leader_joint_7",
                    ArmMsgJointCtrl7,
                    self._codec.decode_170_joint_ctrl_7,
                ),
            }
        )
        return rx
